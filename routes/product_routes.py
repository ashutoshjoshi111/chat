from flask import Blueprint, jsonify

product_blueprint = Blueprint('product', __name__)

@product_blueprint.route('/', methods=['GET'])
def get_products():
    products = [
        {"id": 1, "name": "Laptop"},
        {"id": 2, "name": "Phone"}
    ]
    return jsonify(products)