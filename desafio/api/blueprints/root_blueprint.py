from desafio.api.blueprints.base import BaseController

root = BaseController("root")


@root.route("health", methods=['GET'])
def health_check():
    return dict(aplication_status='ok')


@root.route("resource-status", methods=['GET'])
def resource_status():
    return dict(aplication_status='ok')
