# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:16:13 2025

@author: khush
"""

import os
from smart_attendance import process_attendance

if __name__ == "__main__":
    os.makedirs("./uploaded_group_images", exist_ok=True)
    os.makedirs("./processed", exist_ok=True)

    if os.listdir("./uploaded_group_images"):
        print("\nProcessing attendance...")
        process_attendance()
    else:
        print("No uploaded images found to process")
