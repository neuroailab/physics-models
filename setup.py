from distutils.core import setup

setup(
    name='physion',
    version='1.0',
    packages=['physion'],
    install_requires=[
        'torch',
        'torchvision',
        'h5py',
        'Pillow',
        'clip @ git+https://github.com/openai/CLIP.git@main',
        'timm',
    ]
)
