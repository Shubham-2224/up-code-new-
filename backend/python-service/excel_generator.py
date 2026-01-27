"""
Excel Generator - Create Excel files with extracted voter data
"""

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter
import os
from typing import List, Dict


def generate_excel(data: List[Dict], output_path: str) -> bool:
    """
    Generate Excel file from extracted data
    
    Args:
        data: Array of extracted voter data
        output_path: Path to save Excel file
    
    Returns:
        True if successful, raises exception on error
    """
    try:
        # Create workbook and worksheet
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = 'Voter Data'
        
        # Determine if we should include the image column
        include_images = any(record.get('image_base64') for record in data)
        
        # Define base headers
        headers = [
            'Serial No', 
            'EPIC No', 
            'Name', 'Name (English)', 
            'Relation Type', 
            'Relative Name', 'Relative Name (English)', 
            'House No', 
            'Gender', 
            'Age', 'Assembly No',
            'Booth Center', 'Booth Center (English)',
            'Booth Address', 'Booth Address (English)',
            'Prabhag', 'Booth No'
        ]
        
        if include_images:
            headers.append('Base64 Image String')
        
        # Apply styling to headers
        header_font = Font(bold=True, size=12, color='FFFFFFFF')
        header_fill = PatternFill(start_color='FF4472C4', end_color='FF4472C4', fill_type='solid')
        header_alignment = Alignment(horizontal='center', vertical='center')
 
        worksheet.append(headers) 
 
        # Style header row
        for col_num in range(1, len(headers) + 1):
            cell = worksheet.cell(row=1, column=col_num)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
        
        # Set column widths
        worksheet.column_dimensions['A'].width = 10  # Serial
        worksheet.column_dimensions['B'].width = 15  # EPIC
        worksheet.column_dimensions['C'].width = 25  # Name
        worksheet.column_dimensions['D'].width = 25  # Name (English)
        worksheet.column_dimensions['E'].width = 15  # Relation Type
        worksheet.column_dimensions['F'].width = 25  # Relative Name
        worksheet.column_dimensions['G'].width = 25  # Relative Name (English)
        worksheet.column_dimensions['H'].width = 10  # House
        worksheet.column_dimensions['I'].width = 8   # Gender
        worksheet.column_dimensions['J'].width = 6   # Age
        worksheet.column_dimensions['K'].width = 12  # Assembly
        worksheet.column_dimensions['L'].width = 30  # Booth Center
        worksheet.column_dimensions['M'].width = 30  # Booth Center (Eng)
        worksheet.column_dimensions['N'].width = 30  # Booth Address
        worksheet.column_dimensions['O'].width = 30  # Booth Address (Eng)
        worksheet.column_dimensions['P'].width = 15  # Prabhag
        worksheet.column_dimensions['Q'].width = 15  # Booth No
        
        if include_images:
            worksheet.column_dimensions['R'].width = 25  # Base64
        
        # Define border style
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Sort data by Serial No before writing
        # Use helper to convert Serial No to integer for correct sorting
        def get_serial_key(rec):
            try:
                val = rec.get('serialNo') or rec.get('Serial No') or rec.get('serial_no') or ''
                # Extract only digits in case OCR left artifacts
                digits = ''.join(c for c in str(val) if c.isdigit())
                if digits:
                    return (0, int(digits)) # Priority 0: Numeric Serial
            except:
                pass
            # Fallback to physical order if Serial No is missing or invalid
            return (1, rec.get('page', 0), rec.get('column', 0), rec.get('row', 0))

        data.sort(key=get_serial_key)

        # Add data rows
        for index, record in enumerate(data):
            row_num = index + 2
            
            # Prepare row values in order
            row_values = [
                # Check multiple key variations for Serial No
                record.get('serialNo') or record.get('Serial No') or record.get('serial_no') or '',
                record.get('voterID', ''),
                record.get('name', ''),
                record.get('nameEnglish', ''),
                record.get('relationType', ''),
                record.get('relativeName', ''),
                record.get('relativeNameEnglish', ''),
                record.get('houseNo', ''),
                record.get('gender', ''),
                record.get('age', ''),
                record.get('assemblyNo', ''),
                record.get('boothCenter', ''),
                record.get('boothCenterEnglish', ''),
                record.get('boothAddress', ''),
                record.get('boothAddressEnglish', ''),
                record.get('prabhag', ''),
                record.get('boothNo', '')
            ]

            if include_images:
                row_values.append(record.get('image_base64', ''))
            
            # Write row
            worksheet.append(row_values)
            
            # Style the row
            for col_num in range(1, len(headers) + 1):
                cell = worksheet.cell(row=row_num, column=col_num)
                cell.border = thin_border
                
                cell.alignment = Alignment(vertical='top')

                # Alternate row colors
                if index % 2 == 0:
                    light_gray = PatternFill(start_color='FFF2F2F2', end_color='FFF2F2F2', fill_type='solid')
                    cell.fill = light_gray
        
        # Save workbook
        workbook.save(output_path)
        print(f"Excel file generated: {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Excel generation error: {str(e)}")
        raise

