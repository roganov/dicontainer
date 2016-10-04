import inspect
from abc import ABC, abstractmethod
from collections import deque
from typing import MutableMapping, TypeVar, Generic, Type, Set, List

import pytest  # type: ignore


U = TypeVar('U')


class Container:
    def __init__(self, *configurations):
        self._providers = {}  # type: MutableMapping[Key, Provider]
        binder = Binder()
        for conf in configurations:
            conf(binder)
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

        for key in sorted_keys:
            self._providers[key] = keys_to_bindings[key].create_provider(self._providers)

    def get(self, cls: Type[U]) -> U:
        return self.get_provider(cls).get()

    def get_provider(self, cls: Type[U]) -> 'Provider[U]':
        key = Key(cls)
        providers_map = self._providers
        try:
            provider = providers_map[key]
        except KeyError:
            raise ValueError(key)
        return provider


class DuplicateBindingError(Exception):
    pass


T = TypeVar('T')


class Provider(Generic[T], ABC):
    @abstractmethod
    def get(self) -> T:
        pass


class Binding(ABC):
    linked_to = None  # type: Key


class BindingBuilder(object):
    def __init__(self, interface):
        self.interface = interface
        self._binding = None  # type: Binding

    @property
    def binding(self) -> Binding:
        if self._binding is None:
            return ClassBinding(self.interface, self.interface)
        return self._binding

    def _set_binding(self, binding: Binding):
        if self._binding is not None:
            raise RuntimeError('binding already set')
        self._binding = binding
        return self._binding

    def to(self, impl):
        return self._set_binding(ClassBinding(self.interface, impl))

    def to_instance(self, instance):
        return self._set_binding(InstanceBinding(self.interface, instance))


class ClassBinding(Binding):
    def __init__(self, interface, cls):
        assert inspect.isclass(cls), cls
        assert issubclass(cls, interface), (cls, interface)
        self.interface = interface
        self.cls = cls
        self.key = Key(interface)
        self.linked_to = Key(cls)

    @property
    def dependencies(self) -> 'Set[Key]':
        return set(get_keys_from_constructor(self.cls.__init__).values())

    def create_provider(self, bindings):
        return ClassProvider(self.cls, bindings)


def get_keys_from_constructor(ctor):
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


class ClassProvider(Provider):
    def __init__(self, cls, bindings):
        self.cls = cls
        self.parameters = self._bind_parameters(bindings)

    def _bind_parameters(self, bindings):
        param_names_to_keys = get_keys_from_constructor(self.cls.__init__)
        return {
            name: bindings[key]
            for name, key in param_names_to_keys.items()
        }

    def get(self):
        return self.cls(**{name: provider.get() for name, provider in self.parameters.items()})


class InstanceBinding(Binding):
    def __init__(self, interface, instance):
        assert isinstance is not None
        assert isinstance(instance, interface)
        self.interface = interface
        self.instance = instance
        self.key = Key(interface)

    @property
    def dependencies(self):
        return set()

    def create_provider(self, bindings):
        return InstanceProvider(self.instance)


class InstanceProvider(Provider):
    def __init__(self, instance):
        self.instance = instance

    def get(self):
        return self.instance


class Binder:
    def __init__(self):
        self._binding_builders = []  # type: List[BindingBuilder]

    @property
    def sorted_bindings(self):
        bindings = [b.binding for b in self._binding_builders]
        keys_to_bindings = {}
        for binding in bindings:
            key = binding.key
            if key in keys_to_bindings:
                raise DuplicateBindingError(key)
            keys_to_bindings[key] = binding
        keys_dependencies_graph = {b.key: b.dependencies for b in bindings}
        sorted_keys = topsorted(keys_dependencies_graph)
        return [keys_to_bindings[key] for key in sorted_keys]

    def bind(self, cls):
        builder = BindingBuilder(cls)
        self._binding_builders.append(builder)
        return builder


class Key:
    def __init__(self, interface, annotation=None):
        self.interface = interface
        self.annotation = annotation

    def __eq__(self, other):
        if not isinstance(other, Key):
            return False
        return self.interface is other.interface and self.annotation == other.annotation

    def __hash__(self):
        return hash((self.interface, self.annotation))

    def __repr__(self):
        return '<Key ({})>'.format(self.interface)


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
