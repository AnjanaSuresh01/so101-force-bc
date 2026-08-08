from setuptools import find_packages, setup

package_name = "griff_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anjana Suresh",
    maintainer_email="anjanas222001@gmail.com",
    description="Policy and force-estimator nodes for the SO-101 force-limited stack.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "policy_node = griff_policy.policy_node:main",
            "force_estimator_node = griff_policy.force_estimator_node:main",
        ],
    },
)
