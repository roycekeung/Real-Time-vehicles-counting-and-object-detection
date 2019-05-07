Set objExcel = CreateObject("Excel.Application")
Set objWorkbook = objExcel.Workbooks.Open("D:\HKUST\spring sem\CIEM 5630 Traffic Control Fundamentals and Practice\Project\vision\project optimal green.xlsm")
'objExcel.Workbooks.Open "C:\Program Files\Microsoft Office\Office16\Library\SOLVER\SOLVER.XLAM"


objExcel.Application.Visible = True

objExcel.Application.Run "'D:\HKUST\spring sem\CIEM 5630 Traffic Control Fundamentals and Practice\Project\vision\project optimal green.xlsm'!ThisWorkbook.macro1"

objExcel.ActiveWorkbook.Close

objExcel.Application.Quit
WScript.Echo "Finished."
WScript.Quit