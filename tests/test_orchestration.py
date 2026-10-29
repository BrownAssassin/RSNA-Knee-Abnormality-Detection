import unittest
from types import SimpleNamespace
from orchestrate import push_error


class Orchestration(unittest.TestCase):
    def test_capacity_error_not_success(self):
        response = SimpleNamespace(error='Maximum batch CPU session count of 5 reached.')
        self.assertIn('session count',push_error(response))

    def test_bad_input_not_success(self):
        response = SimpleNamespace(error=None,invalidKernelSources=['missing'])
        self.assertIn('invalidKernelSources',push_error(response))

    def test_valid_push(self):
        self.assertIsNone(push_error(SimpleNamespace(error=None)))


if __name__=='__main__': unittest.main()
