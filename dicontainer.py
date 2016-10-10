import inspect
import threading
from abc import ABC, abstractmethod, abstractproperty
from collections import deque
from threading import Lock
from typing import MutableMapping, TypeVar, Generic, Type, Set, List, Mapping, Hashable, Dict, Callable, Optional, Any, \
    Iterable

import pytest  # type: ignore

T = TypeVar('T')

# TODO:
# - Annotations (names, ...)
# - Refactor to set_binding
# - bind_constant?
# - More configuration options
# - AOP

AnnotationType = Hashable


class Key(Hashable, Generic[T]):
    def __init__(self, interface: Type[T],
                 annotation: AnnotationType=None) -> None:
        self.interface = interface
        self.annotation = annotation

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Key):
            return False
        return self.interface == other.interface and self.annotation == other.annotation

    def __hash__(self) -> int:
        return hash((self.interface, self.annotation))

    def __repr__(self) -> str:
        return '<Key (interface={!r}, annotation={!r})>'.format(
            self.interface, self.annotation)


FunctionConfig = Callable[['Binder'], None]


class Container:
    def __init__(self, *configurations: FunctionConfig) -> None:
        self._providers = {}  # type: MutableMapping[Key, Provider]
        binder = Binder()
        for conf in configurations:
            conf(binder)
        # bind defaults
        binder.bind(Container).to_instance(self)
        binder.bind_scope(noscope, NoScope())
        binder.bind_scope(threadlocal, ThreadlocalScope())
        binder.bind_scope(singleton, SingletonScope())

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
            possible_link = binding.linked_key
            while (possible_link is not None and  # has link
                   key != possible_link and  # not linked to self
                   possible_link in keys_to_bindings  # link has binding
                   ):
                keys_to_bindings[key] = keys_to_bindings[possible_link]
                possible_link = keys_to_bindings[possible_link].linked_key

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


class Scope(ABC):
    @abstractmethod
    def scope(self, key: Key, unscoped: Provider[T]) -> Provider[T]:
        ...

    def __repr__(self) -> str:
        return '<{}>'.format(self.__class__.__name__)


class NoScope(Scope):
    def scope(self, key: Key, unscoped: Provider[T]) -> Provider[T]:
        return unscoped


class ThreadlocalScope(Scope):
    def __init__(self) -> None:
        self._local = threading.local()

    def scope(self, key: Key, unscoped: Provider[T]) -> Provider[T]:
        return ThreadlocalProvider(self._local, key, unscoped)


class ThreadlocalProvider(Provider[T], Generic[T]):
    def __init__(self,
                 threadlocals: threading.local,
                 key: Key,
                 provider: Provider[T]) -> None:
        self._threadlocals = threadlocals
        self._key = key
        self._provider = provider

    def get(self) -> T:
        key = self._key
        try:
            inst = self._threadlocals.__dict__[key]
        except KeyError:
            self._threadlocals.__dict__[key] = inst = self._provider.get()
        return inst


class SingletonScope(Scope):
    def __init__(self) -> None:
        self._providers = {}  # type: Dict[Key, Provider]
        self._lock = Lock()

    def scope(self, key: Key, unscoped: Provider[T]) -> Provider[T]:
        try:
            return self._providers[key]
        except KeyError:
            with self._lock:
                try:
                    return self._providers[key]
                except KeyError:
                    provider = self._providers[key] = InstanceProvider(
                        unscoped.get())
                    return provider


class scope:
    pass


class noscope(scope):
    pass


class threadlocal(scope):
    pass


class singleton(scope):
    pass


class Binding(ABC, Generic[T]):
    @abstractproperty
    def key(self) -> Key[T]:
        ...

    @abstractproperty
    def linked_key(self) -> Optional[Key]:
        ...

    @abstractproperty
    def dependencies(self) -> Set[Key]:
        ...

    @abstractmethod
    def create_provider(self,
                        providers: Mapping[Key, Provider]) -> Provider[T]:
        ...


def provider_key(key: Key[T]) -> Key[Provider[T]]:
    provider_interface = Provider[key.interface]  # type: ignore
    return Key(provider_interface, key.annotation)


class ProviderBinding(Binding[Provider[T]], Generic[T]):
    """
    Internal binding that is responsible for
    binding providers.
    This binding is used whenever Provider[T] is requested.
    """

    def __init__(self, internal_binding: Binding[T]) -> None:
        self._internal_binding = internal_binding
        self._key = provider_key(internal_binding.key)

        if self._internal_binding.linked_key:
            self._linked_key = provider_key(
                self._internal_binding.
                linked_key)  # type: Optional[Key[Provider]]
        else:
            self._linked_key = None

    @property
    def linked_key(self) -> Optional[Key]:
        return self._linked_key

    @property
    def key(self) -> Key[Provider[T]]:
        return self._key

    @property
    def dependencies(self) -> Set[Key]:
        return {self._internal_binding.key}

    def create_provider(
            self, providers: Mapping[Key, Provider]) -> Provider[Provider[T]]:
        return InstanceProvider(providers[self._internal_binding.key])


class ClassBinding(Binding[T], Generic[T]):
    """
    Binding that binds interfaces to concrete classes
    """

    def __init__(self, key: Key[T], cls: Type[T]) -> None:
        assert inspect.isclass(cls), cls
        assert issubclass(cls, key.interface), (cls, key.interface)
        self.cls = cls
        self._key = key
        self._linked_key = Key(cls)
        self._class_injector = ClassInjectorHelper(self.cls)

    @property
    def key(self) -> Key[T]:
        return self._key

    @property
    def linked_key(self) -> Optional[Key]:
        return self._linked_key

    @property
    def dependencies(self) -> Set[Key]:
        return self._class_injector.dependencies

    def create_provider(self,
                        providers: Mapping[Key, Provider]) -> Provider[T]:
        return self._class_injector.create_provider(providers)


class ClassInjectorHelper(Generic[T]):
    def __init__(self, cls: Type[T]) -> None:
        self._cls = cls
        self._param_names_to_keys = get_keys_from_constructor(cls.__init__)

    @property
    def dependencies(self) -> Set[Key]:
        return set(self._param_names_to_keys.values())

    def create_provider(self,
                        providers: Mapping[Key, Provider]) -> Provider[T]:
        param_names_to_providers = {
            name: providers[key]
            for name, key in self._param_names_to_keys.items()
        }
        return ClassProvider(self._cls, param_names_to_providers)


def get_keys_from_constructor(ctor: Any) -> Mapping[str, Key]:
    if ctor is object.__init__:
        return {}
    sig = inspect.signature(ctor)
    param_names_to_keys = {}  # type: Dict[str, Key]
    for parameter_name in list(sig.parameters.keys())[
            1:]:  # skip first param (self)
        parameter = sig.parameters[parameter_name]
        if parameter.annotation is inspect.Parameter.empty:
            raise ValueError('parameter {} has no annotation'.format(
                parameter.name))
        param_names_to_keys[parameter.name] = Key(parameter.annotation)
    return param_names_to_keys


class ClassProvider(Provider[T], Generic[T]):
    def __init__(self, cls: Type[T],
                 parameters: Mapping[str, Provider]) -> None:
        self.cls = cls
        self.parameters = parameters

    def get(self) -> T:
        params = {
            name: provider.get()
            for name, provider in self.parameters.items()
        }
        return self.cls(**params)  # type: ignore


class InstanceBinding(Binding[T], Generic[T]):
    """
    Binding that binds interface to an instance of this interface
    """

    def __init__(self, key: Key, instance: T) -> None:
        assert isinstance is not None
        assert isinstance(instance, key.interface)
        self._instance = instance
        self._key = key

    @property
    def key(self) -> Key[T]:
        return self._key

    @property
    def linked_key(self) -> Optional[Key]:
        return None

    @property
    def dependencies(self) -> Set:
        return set()

    def create_provider(self,
                        providers: Mapping[Key, Provider]) -> Provider[T]:
        return InstanceProvider(self._instance)


class InstanceProvider(Provider[T], Generic[T]):
    def __init__(self, instance: T) -> None:
        self.instance = instance

    def get(self) -> T:
        return self.instance


class ProviderKeyBinding(Binding[T], Generic[T]):
    """
    Binding that binds interface to provider key


    bind(SomeInterface).to_provider(SomeInterfaceProvider)
    """

    def __init__(self, key: Key[T], provider_cls: Type[Provider[T]]) -> None:
        assert isinstance is not None
        assert issubclass(provider_cls, Provider), provider_cls
        self._provider_cls = provider_cls
        self._key = key
        self._class_injector = ClassInjectorHelper(provider_cls)

    @property
    def key(self) -> Key:
        return self._key

    @property
    def linked_key(self) -> Optional[Key]:
        return None

    @property
    def dependencies(self) -> Set[Key]:
        return self._class_injector.dependencies

    def create_provider(self,
                        providers: Mapping[Key, Provider]) -> Provider[T]:
        return ProviderProvider(self._class_injector, providers)


class ProviderProvider(Provider[T], Generic[T]):
    def __init__(self,
                 class_injector: ClassInjectorHelper,
                 providers: Mapping[Key, Provider]) -> None:
        self.provider = class_injector.create_provider(providers)

    def get(self) -> T:
        return self.provider.get().get()


class ScopedBinding(Binding[T], Generic[T]):
    def __init__(self, binding: Binding[T], scope: Scope) -> None:
        self._binding = binding
        self._scope = scope

    @property
    def key(self) -> Key[T]:
        return self._binding.key

    @property
    def linked_key(self) -> Optional[Key]:
        return self._binding.linked_key

    @property
    def dependencies(self) -> Set[Key]:
        return self._binding.dependencies

    def create_provider(self,
                        providers: Mapping[Key, Provider]) -> Provider[T]:
        provider = self._binding.create_provider(providers)
        return self._scope.scope(self.key, provider)

    def __repr__(self) -> str:
        return '<ScopedBinding(binding={!r}, scope={!r})>'.format(
            self._binding, self._scope)


class LazyProvider(Provider[T], Generic[T]):
    def __init__(self) -> None:
        self._underlying_provider = None  # type: Optional[Provider[T]]

    def get(self) -> T:
        provider = self._underlying_provider
        if provider is None:
            raise IllegalStateException
        return provider.get()

    def initialize(self, provider: Provider[T]) -> None:
        assert provider is not None
        self._underlying_provider = provider


class ScopedBindingBuilder(Generic[T]):
    def __init__(self, builder: 'BindingBuilder[T]') -> None:
        self._builder = builder

    def in_scope(self, scope: Type[scope]) -> None:
        self._builder.in_scope(scope)


class LinkedBindingBuilder(ScopedBindingBuilder[T], Generic[T]):
    def to(self, impl: Type[T]) -> ScopedBindingBuilder[T]:
        return self._builder.to(impl)

    def to_instance(self, instance: T) -> None:
        self._builder.to_instance(instance)

    def to_provider(
            self, provider_cls: Type[Provider[T]]) -> ScopedBindingBuilder[T]:
        return self._builder.to_provider(provider_cls)


class AnnotatedBindingBuilder(LinkedBindingBuilder[T], Generic[T]):
    def annotated_with(self,
                       annotation: AnnotationType) -> LinkedBindingBuilder[T]:
        return self._builder.annotated_with(annotation)


class BindingBuilder(Generic[T]):
    def __init__(self, interface: Type[T]) -> None:
        self._key = Key(interface)

        self._scope_type = None  # type: Optional[Type[scope]]

        self._instance = None  # type: Optional[T]
        self._impl = None  # type: Optional[Type[T]]
        self._provider_cls = None  # type: Optional[Type[Provider[T]]]

    def build(self, scopes: Dict[Type[scope], Scope]) -> Binding[T]:
        if self._instance:
            binding = InstanceBinding(self._key,
                                      self._instance)  # type: Binding
        elif self._impl:
            binding = ClassBinding(self._key, self._impl)
        elif self._provider_cls:
            binding = ProviderKeyBinding(self._key, self._provider_cls)
        else:
            # TODO: check if interface is concrete class
            binding = ClassBinding(self._key, self._key.interface)

        if self._scope_type:
            try:
                scope = scopes[self._scope_type]  # type: Optional[Scope]
            except KeyError:
                raise IllegalConfigurationError('scope {!r} not bound'.format(
                    self._scope_type))
        else:
            scope = None

        if scope:
            return ScopedBinding(binding, scope)
        else:
            return binding

    def _check_not_bound(self) -> None:
        if self._instance or self._impl or self._provider_cls:
            raise IllegalConfigurationError('{!r} already bound'.format(
                self._key))

    def annotated_with(self,
                       annotation: AnnotationType) -> LinkedBindingBuilder:
        if self._key.annotation is not None:
            raise RuntimeError(self._key)
        self._key = Key(self._key.interface, annotation)
        return LinkedBindingBuilder(self)

    def in_scope(self, scope: Type[scope]) -> None:
        assert scope is not None
        if self._scope_type is not None:
            raise RuntimeError('scope already set')
        self._scope_type = scope

    def to(self, impl: Type[T]) -> ScopedBindingBuilder[T]:
        assert impl is not None
        self._check_not_bound()
        self._impl = impl
        return ScopedBindingBuilder(self)

    def to_instance(self, instance: T) -> None:
        assert instance is not None
        self._check_not_bound()
        self._instance = instance

    def to_provider(
            self, provider_cls: Type[Provider[T]]) -> ScopedBindingBuilder[T]:
        assert provider_cls is not None
        self._check_not_bound()
        self._provider_cls = provider_cls
        return ScopedBindingBuilder(self)


class Binder:
    def __init__(self) -> None:
        self._binding_builders = []  # type: List[BindingBuilder]
        self._lazy_initialized_providers = {}  # type: Dict[Key, LazyProvider]
        self._scopes = {}  # type: Dict[Type[scope], Scope]

    @property
    def sorted_bindings(self) -> List[Binding]:
        bindings = [b.build(self._scopes) for b in self._binding_builders]
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
            b.key: b.dependencies
            for b in keys_to_bindings.values()
        }
        sorted_keys = topsorted(keys_dependencies_graph)
        try:
            return [keys_to_bindings[key] for key in sorted_keys]
        except KeyError:
            raise KeyNotBoundError(key)

    def bind(self, cls: Type[T]) -> AnnotatedBindingBuilder[T]:
        if issubclass(cls, Provider):
            raise IllegalConfigurationError(
                'cannot bind provider {!r}, bind class and then request provider for this class'
                .format(cls))
        builder = BindingBuilder(cls)
        self._binding_builders.append(builder)
        return AnnotatedBindingBuilder(builder)

    def bind_scope(self, scope_ident: Type[scope], scope_impl: Scope) -> None:
        assert issubclass(scope_ident, scope), scope_ident
        assert isinstance(scope_impl, Scope)
        assert scope_ident not in self._scopes, 'scope {!r} already bound'.format(
            scope_ident)
        self._scopes[scope_ident] = scope_impl

    def get_provider(self, interface: Type[T],
                     annotation: AnnotationType=None) -> Provider[T]:
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


def topsorted(node_to_dependencies: Dict[T, Set[T]]) -> List[T]:
    #  https://algocoding.wordpress.com/2015/04/05/topological-sorting-python/
    node_to_dependants = {}  # type: Dict[T, List[T]]
    for node, deps in node_to_dependencies.items():
        node_to_dependants.setdefault(node, [])
        for dep in deps:
            node_to_dependants.setdefault(dep, []).append(node)
    in_degree = {u: 0 for u in node_to_dependants}  # determine in-degree
    for u in node_to_dependants:  # of each node
        for dependant in node_to_dependants[u]:
            in_degree[dependant] += 1
    in_zero_degrees = deque(
        [n for n, degree in in_degree.items() if degree == 0])

    ordered = []  # list for order of nodes
    while in_zero_degrees:
        u = in_zero_degrees.pop()  # choose node of zero in-degree
        ordered.append(u)
        for dependant in node_to_dependants[u]:
            in_degree[dependant] -= 1
            if in_degree[dependant] == 0:
                in_zero_degrees.appendleft(dependant)

    if len(ordered) == len(node_to_dependants):
        return ordered
    else:  # if there is a cycle,
        raise ValueError(ordered)


def test_bind_self() -> None:
    class A:
        pass

    def configure(binder: Binder) -> None:
        binder.bind(A)

    container = Container(configure)

    a = container.get(A)
    assert isinstance(a, A)
    assert a is not container.get(A)


def test_bind_instance() -> None:
    def configure(binder: Binder) -> None:
        binder.bind(int).to_instance(10)

    assert Container(configure).get(int) == 10


def test_bind_dependency() -> None:
    class A:
        def __init__(self, x: int) -> None:
            self.x = x

    def configure(binder: Binder) -> None:
        binder.bind(int).to_instance(1)
        binder.bind(A)

    a = Container(configure).get(A)

    assert a.x == 1


def test_bind_interface() -> None:
    class A(ABC):
        pass

    class AImpl(A):
        pass

    def configure(binder: Binder) -> None:
        binder.bind(A).to(AImpl)

    container = Container(configure)

    a = container.get(A)
    assert isinstance(a, A)
    assert type(a) is AImpl


def test_late_dependencies_order() -> None:
    class A:
        def __init__(self, x: int) -> None:
            self.x = x

    def configure1(binder: Binder) -> None:
        binder.bind(A)

    def configure2(binder: Binder) -> None:
        binder.bind(int).to_instance(1)

    a = Container(configure1, configure2).get(A)

    assert a.x == 1


def test_linked_binding() -> None:
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

    def configure(binder: Binder) -> None:
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


def test_linked_to_instance() -> None:
    class A:
        pass

    class B(A):
        pass

    class C(B):
        pass

    c = C()

    def configure(binder: Binder) -> None:
        binder.bind(A).to(B)
        binder.bind(B).to_instance(c)

    assert Container(configure).get(A) is c


def test_duplicate_binding() -> None:
    def configure(binder: Binder) -> None:
        binder.bind(int).to_instance(1)
        binder.bind(int).to_instance(2)

    with pytest.raises(DuplicateBindingError):
        Container(configure)


def test_container_provides_self() -> None:
    c = Container()
    assert c.get(Container) is c


def test_provides_provider() -> None:
    def configure(binder: Binder) -> None:
        binder.bind(int).to_instance(1)

    c = Container(configure)

    # mypy doesn't support type aliases for generics
    # https://github.com/python/mypy/issues/606
    ProviderInt = Provider[int]  # type: ignore
    int_provider = c.get(ProviderInt)  # type: ProviderInt
    assert int_provider.get() == 1


def test_key() -> None:
    k1 = Key(List[int])  # type: ignore
    k2 = Key(List[int])  # type: ignore
    assert k1 == k2
    assert hash(k2) == hash(k2)


def test_binder_get_provider() -> None:
    class A:
        provider = None  # type: Provider[int]

    def configure(binder: Binder) -> None:
        provider = binder.get_provider(int)
        with pytest.raises(IllegalStateException):
            provider.get()
        binder.bind(int).to_instance(1)
        a = A()
        a.provider = provider
        binder.bind(A).to_instance(a)

    a = Container(configure).get(A)

    assert a.provider.get() == 1


def test_dependency_on_provider() -> None:
    class A:
        def __init__(self, x: Provider[int]) -> None:
            self.x = x

    def configure(binder: Binder) -> None:
        binder.bind(A)
        binder.bind(int).to_instance(1)

    c = Container(configure)

    assert c.get(A).x.get() == 1


def test_unknown_dependency() -> None:
    class A:
        def __init__(self, x: int) -> None:
            pass

    def configure(binder: Binder) -> None:
        binder.bind(A)

    with pytest.raises(KeyNotBoundError):
        Container(configure)


def test_binding_of_provider_is_disallowed() -> None:
    ProviderInt = Provider[int]  # type: ignore

    def configure(binder: Binder) -> None:
        binder.bind(ProviderInt).to_instance(  # type: ignore
            InstanceProvider(1))

    with pytest.raises(IllegalConfigurationError):
        Container(configure)


def test_bind_concrete_generic() -> None:
    class A(Generic[T]):
        pass

    class AInt(A[int]):
        pass

    AIntInterface = A[int]  # type: ignore

    def configure(binder: Binder) -> None:
        binder.bind(AIntInterface).to(AInt)  # type: ignore

    c = Container(configure)

    assert type(c.get(AIntInterface)) is AInt


def test_bind_to_provider() -> None:
    class A:
        def __init__(self, x: Any) -> None:
            self.x = x

    class AProvider(Provider[A]):
        def __init__(self, x: str) -> None:
            self.x = x

        def get(self) -> A:
            return A(self.x)

    def configure(binder: Binder) -> None:
        binder.bind(str).to_instance('dependency')
        binder.bind(A).to_provider(AProvider)

    a = Container(configure).get(A)

    assert type(a) is A
    assert a.x == 'dependency'


def test_bound_provider_created_every_time_instance_requested() -> None:
    class IntProvider(Provider[int]):
        call_count = 0

        def __init__(self) -> None:
            IntProvider.call_count += 1

        def get(self) -> int:
            return 1

    def configure(binder: Binder) -> None:
        binder.bind(int).to_provider(IntProvider)

    c = Container(configure)
    c.get(int)
    c.get(int)

    assert IntProvider.call_count == 2


def test_no_scope() -> None:
    class A:
        pass

    def configure(binder: Binder) -> None:
        binder.bind(A).in_scope(noscope)

    c = Container(configure)
    assert c.get(A) is not c.get(A)


def test_singleton_scope() -> None:
    class A:
        pass

    def configure(binder: Binder) -> None:
        binder.bind(A).in_scope(singleton)

    c = Container(configure)
    assert c.get(A) is c.get(A)
    assert type(c.get(A)) is A


def test_threadlocal_scope() -> None:
    class A:
        pass

    def configure(binder: Binder) -> None:
        binder.bind(A).in_scope(threadlocal)

    c = Container(configure)

    thread1_a1 = c.get(A)
    thread1_a2 = c.get(A)
    thread2_a1 = None
    thread2_a2 = None

    def init_thread2_a() -> None:
        nonlocal thread2_a1
        nonlocal thread2_a2
        thread2_a1 = c.get(A)
        thread2_a2 = c.get(A)

    t = threading.Thread(target=init_thread2_a)
    t.start()
    t.join()

    assert type(thread1_a1) is A
    assert type(thread2_a1) is A
    assert thread1_a1 is thread1_a2
    assert thread2_a1 is thread2_a2
    assert thread1_a1 is not thread2_a1
