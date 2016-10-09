import inspect
from abc import ABC, abstractmethod, abstractproperty
from collections import deque
from typing import MutableMapping, TypeVar, Generic, Type, Set, List, Any, Mapping, Hashable, Dict

import pytest  # type: ignore


T = TypeVar('T')


class Key(Hashable):
    def __init__(self, interface: Type[Any], annotation=None) -> None:
        self.interface = interface
        self.annotation = annotation

    def __eq__(self, other) -> bool:
        if not isinstance(other, Key):
            return False
        return self.interface == other.interface and self.annotation == other.annotation

    def __hash__(self) -> int:
        return hash((self.interface, self.annotation))

    def __repr__(self) -> str:
        return '<Key (interface={!r}, annotation={!r})>'.format(self.interface, self.annotation)


class Container:
    def __init__(self, *configurations: Any) -> None:
        self._providers = {}  # type: MutableMapping[Key, Provider]
        binder = Binder()
        for conf in configurations:
            conf(binder)
        # bind defaults
        binder.bind(Container).to_instance(self)

        # build keys-bindings map
        sorted_bindings = binder.sorted_bindings
        keys_to_bindings = {}  # type: MutableMapping[Key, Binding]
        sorted_keys = []  # type: List[Key]
        for binding in sorted_bindings:
            keys_to_bindings[binding.key] = binding
            sorted_keys.append(binding.key)

        # collapse linked bindings
        for binding in sorted_bindings:
            key = binding.key
            possible_link = binding.linked_to
            while (
                possible_link is not None and  # has link
                key != possible_link and  # not linked to self
                possible_link in keys_to_bindings  # link has binding
            ):
                keys_to_bindings[key] = keys_to_bindings[possible_link]
                possible_link = keys_to_bindings[possible_link].linked_to

        # build keys-providers map
        for key in sorted_keys:
            provider = keys_to_bindings[key].create_provider(self._providers)
            self._providers[key] = provider

        # initialize lazy providers
        for key, lazy_provider in binder._lazy_initialized_providers.items():
            try:
                provider = self._providers[key]
            except KeyError:
                raise KeyNotBoundError(key)
            lazy_provider.initialize(provider)

    def get(self, cls: Type[T]) -> T:
        return self.get_provider(cls).get()

    def get_provider(self, cls: Type[T]) -> 'Provider[T]':
        key = Key(cls)
        providers_map = self._providers
        try:
            provider = providers_map[key]
        except KeyError:
            raise KeyNotBoundError(key)
        return provider


class DuplicateBindingError(Exception):
    pass


class KeyNotBoundError(Exception):
    pass


class IllegalStateException(Exception):
    pass


class IllegalConfigurationError(Exception):
    pass


class Provider(Generic[T], ABC):
    @abstractmethod
    def get(self) -> T:
        pass


class Binding(ABC):
    linked_to = None  # type: Key

    @abstractproperty
    def key(self) -> Key: ...

    @abstractproperty
    def dependencies(self) -> Set[Key]: ...

    @abstractmethod
    def create_provider(self, providers: Mapping[Key, Provider]) -> Provider: ...


def provider_key(key: Key) -> Key:
    provider_interface = Provider[key.interface]  # type: ignore
    return Key(provider_interface, key.annotation)


class ProviderBinding(Binding):
    """
    Internal binding that is responsible for
    binding providers.
    This binding is used whenever Provider[T] is requested.
    """
    def __init__(self, internal_binding: Binding) -> None:
        self._internal_binding = internal_binding
        self._key = provider_key(internal_binding.key)

        if self._internal_binding.linked_to:
            self._linked_to = provider_key(self._internal_binding.linked_to)
        else:
            self._linked_to = None

    @property
    def linked_to(self):
        return self._linked_to

    @property
    def key(self):
        return self._key

    @property
    def dependencies(self) -> Set[Key]:
        return {self._internal_binding.key}

    def create_provider(self, providers: Mapping[Key, Provider]) -> Provider:
        return InstanceProvider(providers[self._internal_binding.key])


class ClassBinding(Binding):
    """
    Binding that binds interfaces to concrete classes
    """
    def __init__(self, key, cls) -> None:
        assert inspect.isclass(cls), cls
        assert issubclass(cls, key.interface), (cls, key.interface)
        self.cls = cls
        self._key = key
        self.linked_to = Key(cls)
        self._class_injector = ClassInjectorHelper(self.cls)

    @property
    def key(self) -> Key:
        return self._key

    @property
    def dependencies(self) -> Set[Key]:
        return self._class_injector.dependencies

    def create_provider(self, providers: Mapping[Key, Provider]) -> Provider:
        return self._class_injector.create_provider(providers)


class ClassInjectorHelper:
    def __init__(self, cls):
        self._cls = cls
        self._param_names_to_keys = get_keys_from_constructor(cls.__init__)

    @property
    def dependencies(self) -> Set[Key]:
        return set(self._param_names_to_keys.values())

    def create_provider(self, providers: Mapping[Key, Provider]) -> Provider:
        param_names_to_providers = {
            name: providers[key]
            for name, key in self._param_names_to_keys.items()
        }
        return ClassProvider(self._cls, param_names_to_providers)


def get_keys_from_constructor(ctor) -> Mapping[str, Key]:
    if ctor is object.__init__:
        return {}
    sig = inspect.signature(ctor)
    param_names_to_keys = {}
    for parameter_name in list(sig.parameters.keys())[1:]:  # skip first param (self)
        parameter = sig.parameters[parameter_name]
        if parameter.annotation is inspect.Parameter.empty:
            raise ValueError('parameter {} has no annotation'.format(parameter.name))
        param_names_to_keys[parameter.name] = Key(parameter.annotation)
    return param_names_to_keys


class ClassProvider(Provider[T], Generic[T]):
    def __init__(self, cls: Type[T], parameters: Mapping[str, Provider]) -> None:
        self.cls = cls
        self.parameters = parameters

    def get(self) -> T:
        params = {name: provider.get() for name, provider in self.parameters.items()}
        return self.cls(**params)  # type: ignore


class InstanceBinding(Binding):
    """
    Binding that binds interface to an instance of this interface
    """
    def __init__(self, key: Key, instance: T) -> None:
        assert isinstance is not None
        assert isinstance(instance, key.interface)
        self._instance = instance
        self._key = key

    @property
    def key(self) -> Key:
        return self._key

    @property
    def dependencies(self) -> Set:
        return set()

    def create_provider(self, bindings) -> Provider[T]:
        return InstanceProvider(self._instance)


class InstanceProvider(Provider[T], Generic[T]):
    def __init__(self, instance: T) -> None:
        self.instance = instance

    def get(self) -> T:
        return self.instance


class ProviderKeyBinding(Binding):
    """
    Binding that binds interface to provider key


    bind(SomeInterface).to_provider(SomeInterfaceProvider)
    """
    def __init__(self, key: Key, provider_cls: Type[Provider[T]]) -> None:
        assert isinstance is not None
        assert issubclass(provider_cls, Provider), provider_cls
        self._provider_cls = provider_cls
        self._key = key
        self._class_injector = ClassInjectorHelper(provider_cls)

    @property
    def key(self) -> Key:
        return self._key

    @property
    def dependencies(self) -> Set[Key]:
        return self._class_injector.dependencies

    def create_provider(self, providers: Mapping[Key, Provider]) -> Provider[T]:
        return ProviderProvider(self._class_injector, providers)


class ProviderProvider(Provider[T], Generic[T]):
    def __init__(self, class_injector: ClassInjectorHelper, providers: Mapping[Key, Provider]) -> None:
        self.class_injector = class_injector
        self.provider = self.class_injector.create_provider(providers)

    def get(self) -> T:
        return self.provider.get().get()


class LazyProvider(Provider[T], Generic[T]):
    def __init__(self) -> None:
        self._underlying_provider = None  # type: Provider[T]

    def get(self) -> T:
        provider = self._underlying_provider  # type: Provider[T]
        if provider is None:
            raise IllegalStateException
        return provider.get()

    def initialize(self, provider: Provider[T]) -> None:
        assert provider is not None
        self._underlying_provider = provider


class ScopedBindingBuilder:
    def __init__(self, builder: 'BindingBuilder') -> None:
        self._builder = builder

    def in_scope(self, scope) -> None:
        self._builder.in_scope(scope)


class LinkedBindingBuilder(ScopedBindingBuilder):
    def to(self, impl) -> ScopedBindingBuilder:
        return self._builder.to(impl)

    def to_instance(self, instance) -> None:
        self._builder.to_instance(instance)

    def to_provider(self, provider_cls) -> ScopedBindingBuilder:
        return self._builder.to_provider(provider_cls)


class AnnotatedBindingBuilder(LinkedBindingBuilder):
    def annotated_with(self, annotation) -> LinkedBindingBuilder:
        return self._builder.annotated_with(annotation)


class BindingBuilder:
    def __init__(self, interface) -> None:
        self._binding = None  # type: Binding
        self._key = Key(interface)

        # TODO: try to come up with a narrower type
        self._scope = None  # type: Any

        self._instance = None  # type: Any
        self._impl = None  # type: Any
        self._provider_cls = None  # type: Any

    def build(self) -> Binding:
        if self._instance:
            return InstanceBinding(self._key, self._instance)
        if self._impl:
            return ClassBinding(self._key, self._impl)
        if self._provider_cls:
            return ProviderKeyBinding(self._key, self._provider_cls)

        # TODO: check if interface is concrete class
        return ClassBinding(self._key, self._key.interface)

    def _check_not_bound(self):
        if self._instance or self._impl or self._provider_cls:
            raise RuntimeError('{!r} already bound'.format(self._key))

    def _set_binding(self, binding: Binding):
        if self._binding is not None:
            raise RuntimeError('binding already set')
        self._binding = binding

    def annotated_with(self, annotation):
        if self._key.annotation is not None:
            raise RuntimeError(self._key)
        self._key = Key(self._key.interface, annotation)
        return LinkedBindingBuilder(self)

    def in_scope(self, scope):
        assert scope is not None
        if self._scope is not None:
            raise RuntimeError('scope already set')
        self._scope = scope

    def to(self, impl) -> ScopedBindingBuilder:
        assert impl is not None
        self._check_not_bound()
        self._impl = impl
        return ScopedBindingBuilder(self)

    def to_instance(self, instance) -> None:
        assert instance is not None
        self._check_not_bound()
        self._instance = instance

    def to_provider(self, provider_cls) -> ScopedBindingBuilder:
        assert provider_cls is not None
        self._check_not_bound()
        self._provider_cls = provider_cls
        return ScopedBindingBuilder(self)


class Binder:
    def __init__(self) -> None:
        self._binding_builders = []  # type: List[BindingBuilder]
        self._lazy_initialized_providers = {}  # type: Dict[Key, LazyProvider]

    @property
    def sorted_bindings(self) -> List[Binding]:
        bindings = [b.build() for b in self._binding_builders]
        keys_to_bindings = {}  # type: Dict[Key, Binding]
        for binding in bindings:
            key = binding.key
            if key in keys_to_bindings:
                raise DuplicateBindingError(key)
            keys_to_bindings[key] = binding
        # add provider bindings

        for binding in bindings:
            provider_binding = ProviderBinding(binding)
            assert provider_binding.key not in keys_to_bindings, provider_binding
            keys_to_bindings[provider_binding.key] = provider_binding

        keys_dependencies_graph = {
            b.key: b.dependencies for b in keys_to_bindings.values()
        }
        sorted_keys = topsorted(keys_dependencies_graph)
        try:
            return [keys_to_bindings[key] for key in sorted_keys]
        except KeyError:
            raise KeyNotBoundError(key)

    def bind(self, cls) -> AnnotatedBindingBuilder:
        if issubclass(cls, Provider):
            raise IllegalConfigurationError(
                'cannot bind provider {!r}, bind class and then request provider for this class'
                .format(cls))
        builder = BindingBuilder(cls)
        self._binding_builders.append(builder)
        return AnnotatedBindingBuilder(builder)

    def get_provider(self, interface: Type[T], annotation=None) -> Provider[T]:
        """
        May be used to resolve circular dependencies.
        Returns a provider proxy that is initialized upon DI container creation.
        If attempted to request an instance before the container is
        fully constructed, the return provider
        will raise `IllegalStateException` exception.
        """
        key = Key(interface, annotation)
        try:
            provider = self._lazy_initialized_providers[key]
        except KeyError:
            provider = self._lazy_initialized_providers[key] = LazyProvider()
        return provider


def topsorted(node_to_dependencies):
    #  https://algocoding.wordpress.com/2015/04/05/topological-sorting-python/
    node_to_dependants = {}
    for node, deps in node_to_dependencies.items():
        node_to_dependants.setdefault(node, [])
        for dep in deps:
            node_to_dependants.setdefault(dep, []).append(node)
    in_degree = {u: 0 for u in node_to_dependants}     # determine in-degree
    for u in node_to_dependants:                          # of each node
        for dependant in node_to_dependants[u]:
            in_degree[dependant] += 1
    in_zero_degrees = deque([n for n, degree in in_degree.items() if degree == 0])

    ordered = []     # list for order of nodes
    while in_zero_degrees:
        u = in_zero_degrees.pop()          # choose node of zero in-degree
        ordered.append(u)
        for dependant in node_to_dependants[u]:
            in_degree[dependant] -= 1
            if in_degree[dependant] == 0:
                in_zero_degrees.appendleft(dependant)

    if len(ordered) == len(node_to_dependants):
        return ordered
    else:                    # if there is a cycle,
        raise ValueError(ordered)


def test_bind_self():
    class A:
        pass

    def configure(binder):
        binder.bind(A)

    container = Container(configure)

    a = container.get(A)
    assert isinstance(a, A)
    assert a is not container.get(A)


def test_bind_instance():
    def configure(binder):
        binder.bind(int).to_instance(10)

    assert Container(configure).get(int) == 10


def test_bind_dependency():
    class A:
        def __init__(self, x: int) -> None:
            self.x = x

    def configure(binder):
        binder.bind(int).to_instance(1)
        binder.bind(A)

    a = Container(configure).get(A)

    assert a.x == 1


def test_bind_interface():
    class A(ABC):
        pass

    class AImpl(A):
        pass

    def configure(binder):
        binder.bind(A).to(AImpl)

    container = Container(configure)

    a = container.get(A)
    assert isinstance(a, A)
    assert type(a) is AImpl


def test_late_dependencies_order():
    class A:
        def __init__(self, x: int) -> None:
            self.x = x

    def configure1(binder):
        binder.bind(A)

    def configure2(binder):
        binder.bind(int).to_instance(1)

    a = Container(configure1, configure2).get(A)

    assert a.x == 1


def test_linked_binding():
    class A:
        pass

    class B(A):
        pass

    class C(B):
        pass

    class D(C):
        pass

    class Dep:
        def __init__(self, a: A) -> None:
            self.a = a

    def configure(binder):
        binder.bind(A).to(B)
        binder.bind(B).to(C)
        binder.bind(C).to(D)
        binder.bind(Dep)

    container = Container(configure)

    # direct links
    assert type(container.get(A)) is D
    assert type(container.get(B)) is D

    # providers links
    assert type(container.get(Provider[A]).get()) is D  # type: ignore
    assert type(container.get(Provider[B]).get()) is D  # type: ignore

    # dependency link
    assert type(container.get(Dep).a) is D


def test_linked_to_instance():
    class A:
        pass
    class B(A):
        pass
    class C(B):
        pass

    c = C()

    def configure(binder):
        binder.bind(A).to(B)
        binder.bind(B).to_instance(c)

    assert Container(configure).get(A) is c


def test_duplicate_binding():
    def configure(binder):
        binder.bind(int).to_instance(1)
        binder.bind(int).to_instance(2)

    with pytest.raises(DuplicateBindingError):
        Container(configure)


def test_container_provides_self():
    c = Container()
    assert c.get(Container) is c


def test_provides_provider():
    def configure(binder):
        binder.bind(int).to_instance(1)

    c = Container(configure)

    int_provider = c.get(Provider[int])  # type: ignore
    assert int_provider.get() == 1


def test_key():
    k1 = Key(List[int])  # type: ignore
    k2 = Key(List[int])  # type: ignore
    assert k1 == k2
    assert hash(k2) == hash(k2)


def test_binder_get_provider():
    class A:
        pass

    def configure(binder):
        provider = binder.get_provider(int)
        with pytest.raises(IllegalStateException):
            provider.get()
        binder.bind(int).to_instance(1)
        a = A()
        a.provider = provider
        binder.bind(A).to_instance(a)

    a = Container(configure).get(A)

    assert a.provider.get() == 1


def test_dependency_on_provider():
    class A:
        def __init__(self, x: Provider[int]) -> None:
            self.x = x

    def configure(binder: Binder):
        binder.bind(A)
        binder.bind(int).to_instance(1)

    c = Container(configure)

    assert c.get(A).x.get() == 1


def test_unknown_dependency():
    class A:
        def __init__(self, x: int) -> None:
            pass

    def configure(binder):
        binder.bind(A)

    with pytest.raises(KeyNotBoundError):
        Container(configure)


def test_binding_of_provider_is_disallowed():
    ProviderInt = Provider[int]  # type: ignore

    def configure(binder):
        binder.bind(ProviderInt).to_instance(InstanceProvider(1))

    with pytest.raises(IllegalConfigurationError):
        Container(configure)


def test_bind_concrete_generic():
    class A(Generic[T]):
        pass

    class AInt(A[int]):
        pass

    AIntInterface = A[int]  # type: ignore

    def configure(binder):
        binder.bind(AIntInterface).to(AInt)

    c = Container(configure)

    assert type(c.get(AIntInterface)) is AInt


def test_bind_to_provider():
    class A:
        def __init__(self, x):
            self.x = x

    class AProvider(Provider[A]):
        def __init__(self, x: str) -> None:
            self.x = x

        def get(self) -> A:
            return A(self.x)

    def configure(binder):
        binder.bind(str).to_instance('dependency')
        binder.bind(A).to_provider(AProvider)

    a = Container(configure).get(A)

    assert type(a) is A
    assert a.x == 'dependency'


def test_bound_provider_created_every_time_instance_requested():
    class IntProvider(Provider[int]):
        call_count = 0

        def __init__(self):
            IntProvider.call_count += 1

        def get(self) -> int:
            return 1

    def configure(binder: Binder):
        binder.bind(int).to_provider(IntProvider)

    c = Container(configure)
    c.get(int)
    c.get(int)

    assert IntProvider.call_count == 2
