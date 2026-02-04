#!/usr/bin/env python3
"""
Quick fix script for common issues
"""

import os
import sys

def check_files():
    """Check if all required files are present"""
    required_files = [
        'main_mvp.py',
        'config_mvp.py',
        'classifier_mvp.py',
        'analytics_dashboard.py'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("❌ Missing files:")
        for file in missing:
            print(f"   - {file}")
        print("\n💡 Please download all files from the outputs folder.")
        return False
    
    print("✅ All required files present")
    return True


def fix_imports():
    """Fix import statements in main_mvp.py"""
    try:
        with open('main_mvp.py', 'r') as f:
            content = f.read()
        
        # Check if imports are correct
        if 'from config import' in content:
            print("⚠️  Found incorrect import: 'from config import'")
            print("   Fixing to: 'from config_mvp import'")
            
            content = content.replace('from config import', 'from config_mvp import')
            
            with open('main_mvp.py', 'w') as f:
                f.write(content)
            
            print("✅ Fixed imports in main_mvp.py")
            return True
        else:
            print("✅ Imports are already correct")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing imports: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'ultralytics': 'ultralytics',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True


def main():
    print("\n" + "="*60)
    print("🔧 FIX COMMON ISSUES")
    print("="*60 + "\n")
    
    # Check 1: Files
    print("1. Checking files...")
    if not check_files():
        sys.exit(1)
    
    # Check 2: Dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check 3: Fix imports
    print("\n3. Checking imports...")
    if not fix_imports():
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ ALL CHECKS PASSED!")
    print("="*60)
    print("\nYou can now run:")
    print("  python main_mvp.py --video your_video.mp4")
    print("\n")


if __name__ == "__main__":
    main()