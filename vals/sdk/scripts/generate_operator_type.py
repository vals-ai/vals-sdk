"""
Internal script to regenerate the operator_type.py file.

Run with python generate_operator_type.py.
"""

import asyncio

from vals.sdk.util import get_ariadne_client


async def generate_operator_type():
    client = get_ariadne_client()
    with open("operator_type_template", "r") as template:
        template_text = template.read()

    operator_list = await client.get_operators()
    list_of_names = [f'"{op.name_in_doc}"' for op in operator_list.operators]

    template_text = template_text.replace("<REPLACE>", ",\n\t".join(list_of_names))

    with open("operator_type.py", "w") as f:
        f.write(template_text)


if __name__ == "__main__":
    asyncio.run(generate_operator_type())
