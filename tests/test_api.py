import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app import app

@pytest.fixture
def client():
    """
    This pytest fixture sets up a test client for the Flask application.
    It provides a way for tests to interact with the Flask app in a controlled environment.
    The 'yield' statement returns the test client which can then be used by test functions.
    After the test function completes, any cleanup can occur after the 'yield' statement.
    """
    with app.test_client() as client:
        yield client

def test_recommend_endpoint(client):
    """
    This function tests the recommendation endpoint of the Flask application.
    It uses the test client provided by the 'client' fixture to make a GET request to the '/recommend/{user_id}' endpoint.
    
    Args:
    - client: The Flask test client instance to simulate requests to the application.

    It checks if:
    - The status code of the response is 200, indicating success.
    - (You can extend this test to check more conditions, such as the content of the response, to ensure the endpoint behaves as expected.)
    """
    test_user_id = '360' #this is an arbitrary user id
    response = client.get(f'/recommend/{test_user_id}')
    assert response.status_code == 200