from setuptools import setup

package_name = 'my_vision_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sutd',
    maintainer_email='sutd@todo.todo',
    description='Mission and visualization package',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'visualizer_node = my_vision_pkg.object_visualizer:main',
            'mission_node = my_vision_pkg.mission_node:main',
        ],
    },
)
