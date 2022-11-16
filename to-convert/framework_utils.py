from fastapi.openapi.utils import get_openapi


def custom_docs(app, title, version, description, root_path, exmaples_info: list):
    def custom_openapi():
        """
        ref: https://fastapi.tiangolo.com/advanced/extending-openapi/
        """
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=title,
            version=version,
            description=description,
            routes=app.routes,
        )

        # doing this to fix the paths in the docs. ref: https://fastapi.tiangolo.com/advanced/behind-a-proxy/
        openapi_schema["servers"] = [{"url": root_path}]  # this should be same as the class name

        def add_examples_docs(end_point, req_type, param_index, description, example_vals: list):
            openapi_schema["paths"][end_point][req_type]["parameters"][param_index]["description"] = description
            examples = {}
            for val in example_vals:
                examples[str(val)] = {"value": val}
            openapi_schema["paths"][end_point][req_type]["parameters"][param_index]["examples"] = examples

        # add examples for auto generated API docs
        for exmaple_info in exmaples_info:
            add_examples_docs(*exmaple_info)

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    return custom_openapi
