#!/usr/bin/env python3
"""
Automatic dependency installer for Movie Recommendation System
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {command}")
            return True
        else:
            print(f"❌ {command}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {command} - Exception: {e}")
        return False

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_basic_packages():
    """Install basic required packages"""
    print("\n📦 Installing basic packages...")
    
    packages = [
        'flask',
        'flask-sqlalchemy', 
        'flask-bcrypt',
        'flask-cors',
        'numpy',
        'pandas',
        'scikit-learn',
        'requests'
    ]
    
    success_count = 0
    
    for package in packages:
        print(f"\nInstalling {package}...")
        if run_command(f"pip install {package}"):
            success_count += 1
        else:
            # Try alternative methods
            print(f"Trying alternative installation for {package}...")
            if run_command(f"python -m pip install {package}"):
                success_count += 1
            elif run_command(f"pip3 install {package}"):
                success_count += 1
    
    print(f"\n📊 Successfully installed {success_count}/{len(packages)} packages")
    return success_count == len(packages)

def install_optional_packages():
    """Install optional packages for enhanced features"""
    print("\n🎯 Installing optional packages...")
    
    optional_packages = [
        'tqdm',
        'matplotlib', 
        'seaborn'
    ]
    
    for package in optional_packages:
        print(f"Installing {package}...")
        run_command(f"pip install {package}")

def test_imports():
    """Test if packages can be imported"""
    print("\n🧪 Testing imports...")
    
    test_packages = {
        'flask': 'Flask',
        'flask_sqlalchemy': 'Flask-SQLAlchemy', 
        'flask_bcrypt': 'Flask-Bcrypt',
        'flask_cors': 'Flask-CORS',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'requests': 'Requests'
    }
    
    success_count = 0
    
    for module, name in test_packages.items():
        try:
            __import__(module)
            print(f"✅ {name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name} - {e}")
    
    print(f"\n📊 Successfully imported {success_count}/{len(test_packages)} packages")
    return success_count == len(test_packages)

def create_simple_requirements():
    """Create a simple requirements.txt file"""
    print("\n📄 Creating requirements.txt...")
    
    requirements = """flask==2.3.3
flask-sqlalchemy==3.0.5
flask-bcrypt==1.0.1
flask-cors==4.0.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
requests==2.31.0
tqdm==4.66.1
matplotlib==3.7.2
seaborn==0.12.2
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ requirements.txt created")

def main():
    """Main installation function"""
    print("🎬 Movie Recommendation System - Dependency Installer")
    print("=" * 60)
    
    # Check Python version
    if not check_python():
        print("❌ Please upgrade to Python 3.8 or higher")
        return False
    
    # Create requirements file
    create_simple_requirements()
    
    # Try to install from requirements.txt first
    print("\n📦 Trying to install from requirements.txt...")
    if run_command("pip install -r requirements.txt"):
        print("✅ All packages installed from requirements.txt")
    else:
        print("❌ requirements.txt installation failed, trying individual packages...")
        install_basic_packages()
    
    # Install optional packages
    install_optional_packages()
    
    # Test imports
    if test_imports():
        print("\n🎉 All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Run: python test_setup.py")
        print("2. If tests pass, run: python start.py")
        print("3. Open browser: http://localhost:5000")
        return True
    else:
        print("\n❌ Some dependencies failed to install.")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection")
        print("2. Try running as administrator/sudo")
        print("3. Update pip: python -m pip install --upgrade pip")
        print("4. Use virtual environment")
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)