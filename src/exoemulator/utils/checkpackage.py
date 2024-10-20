def check_installed(package_name):
    import pkg_resources

    installed_packages = [pkg.key for pkg in pkg_resources.working_set]

    if package_name in installed_packages:
        print(f"{package_name} is installed")
    else:
        raise ValueError(f"{package_name} is not installed")
