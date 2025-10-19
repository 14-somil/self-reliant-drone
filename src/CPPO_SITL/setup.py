from setuptools import find_packages, setup

package_name = 'CPPO_SITL'

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
    maintainer='Somil Agrawal',
    maintainer_email='somil0014@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test = CPPO_SITL.test_script:main',
            'train = CPPO_SITL.train:main'
        ],
    },
)
