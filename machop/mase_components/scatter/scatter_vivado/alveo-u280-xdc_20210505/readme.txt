*******************************************************************************
** © Copyright 2021 Xilinx, Inc. All rights reserved.
** This file contains confidential and proprietary information of Xilinx, Inc. and 
** is protected under U.S. and international copyright and other intellectual property laws.
*******************************************************************************
**   ____  ____ 
**  /   /\/   / 
** /___/  \  /   Vendor: Xilinx 
** \   \   \/    
**  \   \        readme.txt Version: 2.04  
**  /   /        Date Last Modified:  09DEC2020
** /___/   /\    Date Created: 06AUG2019
** \   \  /  \   Associated Filename: alveo-u280-xdc.zip
**  \___\/\___\ 
** 
**  Device: Alveo XCU280 Production Release
**  Purpose: XDC constraints file
**  Reference: 
**   
*******************************************************************************
**
**  Disclaimer: 
**
**		This disclaimer is not a license and does not grant any rights to the materials 
**              distributed herewith. Except as otherwise provided in a valid license issued to you 
**              by Xilinx, and to the maximum extent permitted by applicable law: 
**              (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, 
**              AND XILINX HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, 
**              INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR 
**              FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether in contract 
**              or tort, including negligence, or under any other theory of liability) for any loss or damage 
**              of any kind or nature related to, arising under or in connection with these materials, 
**              including for any direct, or any indirect, special, incidental, or consequential loss 
**              or damage (including loss of data, profits, goodwill, or any type of loss or damage suffered 
**              as a result of any action brought by a third party) even if such damage or loss was 
**              reasonably foreseeable or Xilinx had been advised of the possibility of the same.


**  Critical Applications:
**
**		Xilinx products are not designed or intended to be fail-safe, or for use in any application 
**		requiring fail-safe performance, such as life-support or safety devices or systems, 
**		Class III medical devices, nuclear facilities, applications related to the deployment of airbags,
**		or any other applications that could lead to death, personal injury, or severe property or 
**		environmental damage (individually and collectively, "Critical Applications"). Customer assumes 
**		the sole risk and liability of any use of Xilinx products in Critical Applications, subject only 
**		to applicable laws and regulations governing limitations on product liability.

**  THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

*******************************************************************************


*******************************************************************************

** IMPORTANT NOTES **

1. REVISION HISTORY 

            Readme  
Date        Version      Revision Description
=========================================================================
06AUG2019   1.0          Initial Xilinx release for Pre Poduction A-U280 ES1.
03FEB2020   2.0          Initial Xilinx release for Production A-U280.
12FEB2020   2.01         Fixed text typos.
24FEB2020   2.02         Corrected PN typos on clocks and DIFF_SSTL12_DCI on DDR4 clocks.
13MAY2020   2.03         Added Bitstream generation constraints.
09DEC2020   2.04         Changed config rate to 63.8Mhz.
=========================================================================



2. DESIGN FILE HIERARCHY

\alveo-u280-xdc.xdc
