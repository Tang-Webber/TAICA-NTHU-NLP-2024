import pkg_resources

def get_package_versions():
    packages = [
        'numpy',
        'pandas', 
        'torch',
        'matplotlib',
        'seaborn',
        'opencc',
        'scikit-learn',
        'tqdm'
    ]
    
    requirements = []
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            requirements.append(f"{package}=={version}")
        except pkg_resources.DistributionNotFound:
            print(f"Warning: {package} not found")
    
    return requirements

def write_requirements():
    requirements = get_package_versions()
    
    with open('requirements.txt', 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
            print(f"Added: {req}")

if __name__ == "__main__":
    write_requirements()
