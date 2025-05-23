from setuptools import find_packages, setup

package_name = 'px4_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='plague',
    maintainer_email='plague@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bench_mark = px4_nav.bench_mark:main',
            'error_calculator = px4_nav.error_calculator:main',
            'path_efficiency = px4_nav.path_efficiency:main',
            'yaw_plotter = px4_nav.yaw_plotter:main',
            'gz_ros_bridge = px4_nav.gz_ros_bridge:main',
            'live_position_plotter = px4_nav.live_position_plotter:main',
            'image_translator = px4_nav.image_translator:main',
            'optical_flow = px4_nav.optical_flow:main',
        ],
    },
)
