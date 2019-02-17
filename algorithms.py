from operator import *
from functools import *


def compose(function, *functions):
    '''
    conpose functions from left to right
    '''

    return lambda *args: compose(*functions)(function(*args)) if functions else function(*args)


def map_innermost_element(function, sequence, classes=(list,)):
    '''
    apply function to innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (type(sequence)(map(lambda element: map_innermost_element(function, element, classes=classes), sequence))
            if isinstance(sequence, classes) else function(sequence))


def map_innermost_list(function, sequence, classes=(list,)):
    '''
    apply function to innermost lists.
    innermost list is defined as list which doesn't contain instance of "classes" (default: list)
    '''

    return (type(sequence)(map(lambda element: map_innermost_list(function, element, classes=classes), sequence))
            if isinstance(sequence, classes) and any(map(lambda element: isinstance(element, classes), sequence)) else function(sequence))


def enumerate_innermost_element(sequence, classes=(list,), indices=()):
    '''
    make tuple of innermost element and index.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (type(sequence)(map(lambda index_element: enumerate_innermost_element(index_element[1], classes=classes, indices=indices + (index_element[0],)), enumerate(sequence)))
            if isinstance(sequence, classes) else (indices, sequence))


def enumerate_innermost_list(sequence, classes=(list,), indices=()):
    '''
    make tuple of innermost element and index.
    innermost list is defined as list which doesn't contain instance of "classes" (default: list)
    '''

    return (type(sequence)(map(lambda index_element: enumerate_innermost_list(index_element[1], classes=classes, indices=indices + (index_element[0],)), enumerate(sequence)))
            if isinstance(sequence, classes) and any(map(lambda element: isinstance(element, classes), sequence)) else (indices, sequence))


def zip_innermost_element(*sequences, classes=(list,)):
    '''
    make tuple of innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (list(map(lambda elements: zip_innermost_element(*elements, classes=classes), zip(*sequences)))
            if all(map(lambda sequence: isinstance(sequence, classes), sequences)) else sequences)


def zip_innermost_list(*sequences, classes=(list,)):
    '''
    make tuple of innermost elements.
    innermost list is defined as list which doesn't contain instance of "classes" (default: list)
    '''

    return (list(map(lambda elements: zip_innermost_list(*elements, classes=classes), zip(*sequences)))
            if all(map(lambda sequence: isinstance(sequence, classes) and any(map(lambda element: isinstance(element, classes), sequence)), sequences)) else sequences)


def flatten_innermost_element(sequence, classes=(list,)):
    '''
    return flattened list of innermost elements.
    innermost element is defined as element which is not instance of "classes" (default: list)
    '''

    return (reduce(add, map(lambda element: flatten_innermost_element(element, classes=classes), sequence), [])
            if isinstance(sequence, classes) else [sequence])


def flatten_innermost_list(sequence, classes=(list,)):
    '''
    return flattened list of innermost elements.
    innermost list is defined as list which doesn't contain instance of "classes" (default: list)
    '''

    return (reduce(add, map(lambda element: flatten_innermost_list(element, classes=classes), sequence))
            if isinstance(sequence, classes) and any(map(lambda element: isinstance(element, classes), sequence)) else [sequence])
