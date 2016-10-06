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
            # add provider for key
            provider = keys_to_bindings[key].create_provider(self._providers)
            self._providers[key] = provider
            # also add provider for provider for key (huh!)
            provider_interface = Provider[key.interface]  # type: ignore
            provider_key = Key(provider_interface, key.annotation)

            # TODO: is this correct that silently don't do anything
            # if provider for such key was already provided by user?
            if provider_key not in self._providers:
                self._providers[provider_key] = InstanceProvider(provider)

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


class Provider(Generic[T], ABC):
    @abstractmethod
    def get(self) -> T:
        pass


class Binding(ABC):
    linked_to = None  # type: Key

    @abstractproperty
    def key(self) -> Key: ...

    @abstractproperty
    def dependencies(self) -> Set: ...

    @abstractmethod
    def create_provider(self, providers: Mapping[Key, Provider]) -> Provider:
        pass


class ClassBinding(Binding):
    def __init__(self, key, cls) -> None:
        assert inspect.isclass(cls), cls
        assert issubclass(cls, key.interface), (cls, key.interface)
        self.cls = cls
        self._key = key
        self.linked_to = Key(cls)

    @property
    def key(self) -> Key:
        return self._key

    @property
    def dependencies(self) -> Set[Key]:
        return set(get_keys_from_constructor(self.cls.__init__).values())

    def create_provider(self, providers: Mapping[Key, Provider]):
        return ClassProvider(self.cls, providers)


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
    def __init__(self, cls: Type[T], providers) -> None:
        self.cls = cls
        self.parameters = self._bind_parameters(providers)

    def _bind_parameters(self, providers):
        param_names_to_keys = get_keys_from_constructor(self.cls.__init__)
        return {
            name: providers[key]
            for name, key in param_names_to_keys.items()
        }

    def get(self) -> T:
        params = {name: provider.get() for name, provider in self.parameters.items()}
        return self.cls(**params)  # type: ignore


class InstanceBinding(Binding):
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


class AnnotatedBindingBuilder(LinkedBindingBuilder):
    def annotated_with(self, annotation) -> LinkedBindingBuilder:
        return self._builder.annotated_with(annotation)


class BindingBuilder(object):
    def __init__(self, interface) -> None:
        self._binding = None  # type: Binding
        self._key = Key(interface)
        self._scope = None

        self._instance = None
        self._impl = None

    @property
    def binding(self) -> Binding:
        if self._instance:
            return InstanceBinding(self._key, self._instance)
        if self._impl:
            return ClassBinding(self._key, self._impl)

        # TODO: check if interface is concrete class
        return ClassBinding(self._key, self._key.interface)

    def _check_not_bound(self):
        if self._instance or self._impl:
            raise RuntimeError('already bound')

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
        self._check_not_bound()
        self._impl = impl
        return ScopedBindingBuilder(self)

    def to_instance(self, instance) -> None:
        self._check_not_bound()
        self._instance = instance


class Binder:
    def __init__(self) -> None:
        self._binding_builders = []  # type: List[BindingBuilder]
        self._lazy_initialized_providers = {}  # type: Dict[Key, LazyProvider]

    @property
    def sorted_bindings(self) -> List[Binding]:
        bindings = [b.binding for b in self._binding_builders]
        keys_to_bindings = {}  # type: Dict[Key, Binding]
        for binding in bindings:
            key = binding.key
            if key in keys_to_bindings:
                raise DuplicateBindingError(key)
            keys_to_bindings[key] = binding
        keys_dependencies_graph = {b.key: b.dependencies for b in bindings}
        sorted_keys = topsorted(keys_dependencies_graph)
        return [keys_to_bindings[key] for key in sorted_keys]

    def bind(self, cls) -> AnnotatedBindingBuilder:
        builder = BindingBuilder(cls)
        self._binding_builders.append(builder)
        return AnnotatedBindingBuilder(builder)

    def get_provider(self, interface: Type[T], annotation=None) -> Provider[T]:
        """
        May be used to resolve circular dependencies.
        Returns a provider proxy that is initialized upon DI container creation.
        If attempted to request an instance before the container is
        constructed, the provider with raise `IllegalStateException` exception.
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

    def configure(binder):
        binder.bind(A).to(B)
        binder.bind(B).to(C)
        binder.bind(C).to(D)

    container = Container(configure)
    assert type(container.get(A)) is D
    assert type(container.get(B)) is D


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
