# NOTE: Mock Server Tests from: https://gist.github.com/eruvanos/f6f62edb368a20aaa880e12976620db8

import unittest
import requests

from .server_mock import MockServer

class TestMockServer(unittest.TestCase):
    def setUp(self):
        self.server = MockServer(port=1234)
        self.server.start()

    def test_mock_with_callback(self):
        self.called = False

        def callback():
            self.called = True
            return 'Hallo'

        self.server.add_callback_response("/callback", callback)

        response = requests.get(self.server.url + "/callback")

        self.assertEqual(200, response.status_code)
        self.assertEqual('Hallo', response.text)
        self.assertTrue(self.called)

    def tearDown(self):
        self.server.shutdown_server()