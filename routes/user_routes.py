from flask import Blueprint, jsonify, request

user_blueprint = Blueprint('user', __name__)


# Define a sample GET endpoint
@user_blueprint.route('/', methods=['GET'])
def get_users():
    users = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"}
    ]
    return jsonify(users)


# Define a POST endpoint
@user_blueprint.route('/', methods=['POST'])
def create_user():
    data = request.get_json()
    new_user = {"id": data["id"], "name": data["name"]}
    return jsonify(new_user), 201
