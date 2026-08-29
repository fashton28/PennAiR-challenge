from setuptools import find_packages, setup

package_name = "pennair_vision"

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
    maintainer="Fabian Ashton",
    maintainer_email="fashton502@gmail.com",
    description="Video streaming and shape detection nodes for the PennAiR challenge.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "video_publisher = pennair_vision.video_publisher:main",
            "shape_detector = pennair_vision.shape_detector:main",
        ],
    },
)
