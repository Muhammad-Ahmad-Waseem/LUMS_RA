# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import pkgutil
import os
import sys

import rasterio

datas = collect_data_files("skimage.io._plugins")
hiddenimports = collect_submodules('skimage.io._plugins')

for package in pkgutil.iter_modules(rasterio.__path__, prefix="rasterio."):
    hiddenimports.append(package.name)

hiddenimports.append('pywt._extensions._cwt')
hiddenimports.append(r'D:\LUMS_RA\Codes\Python_Codes\Built_UnBuilt_with_GUI\model.py')
hiddenimports.append(r'D:\LUMS_RA\Codes\Python_Codes\Built_UnBuilt_with_GUI\decoder.py')

datas  = []
datas += collect_data_files('timm', include_py_files=True)

datas.append(('D:\LUMS_RA\Codes\Python_Codes\Built_UnBuilt_with_GUI\App Logo.ico', '.'))
datas.append(('D:\LUMS_RA\Codes\Python_Codes\Built_UnBuilt_with_GUI\Background.png', '.'))

block_cipher = None


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
