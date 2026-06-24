@echo off
REM zx-work-rag Pipeline Runner
REM Usage: run_pipeline.bat [step]
REM Steps: scan, dedup, identify, convert, extract, embed, all

setlocal
set ROOT=%~dp0
set PYTHON=%ROOT%.venv\Scripts\python.exe
set SCRIPTS=%ROOT%scripts

if "%1"=="" goto usage
if "%1"=="scan" goto scan
if "%1"=="dedup" goto dedup
if "%1"=="identify" goto identify
if "%1"=="convert" goto convert
if "%1"=="extract" goto extract
if "%1"=="embed" goto embed
if "%1"=="all" goto all
if "%1"=="web" goto web
if "%1"=="stats" goto stats
goto usage

:scan
echo [Step 1/6] Scanning files...
%PYTHON% %SCRIPTS%\01_scan_files.py
goto end

:dedup
echo [Step 2/6] Deduplicating...
%PYTHON% %SCRIPTS%\02_dedup.py
goto end

:identify
echo [Step 3/6] Identifying file types...
%PYTHON% %SCRIPTS%\03_identify_types.py
goto end

:convert
echo [Step 4/6] Converting old formats...
%PYTHON% %SCRIPTS%\04_convert_formats.py
goto end

:extract
echo [Step 5/6] Extracting text...
%PYTHON% %SCRIPTS%\05_extract_text.py
goto end

:embed
echo [Step 6/6] Generating embeddings...
%PYTHON% %SCRIPTS%\06_embed_cloud.py
goto end

:all
echo Running full pipeline...
echo [Step 1/6] Scanning files...
%PYTHON% %SCRIPTS%\01_scan_files.py
echo [Step 2/6] Deduplicating...
%PYTHON% %SCRIPTS%\02_dedup.py
echo [Step 3/6] Identifying file types...
%PYTHON% %SCRIPTS%\03_identify_types.py
echo [Step 4/6] Converting old formats...
%PYTHON% %SCRIPTS%\04_convert_formats.py
echo [Step 5/6] Extracting text...
%PYTHON% %SCRIPTS%\05_extract_text.py
echo [Step 6/6] Generating embeddings...
%PYTHON% %SCRIPTS%\06_embed_cloud.py
echo Pipeline complete!
goto end

:web
echo Starting Streamlit web app...
%ROOT%.venv\Scripts\streamlit.exe run %ROOT%server\web_app.py --server.port 8501
goto end

:stats
%PYTHON% -c "import sys; sys.path.insert(0, r'%ROOT%'); from server.rag_query import RAGQueryService; svc = RAGQueryService('none'); import json; print(json.dumps(svc.get_stats(), indent=2)); svc.close()"
goto end

:usage
echo Usage: run_pipeline.bat [step]
echo Steps:
echo   scan      - Scan all files in source directory
echo   dedup     - Deduplicate by MD5 hash
echo   identify  - Identify types for extensionless files
echo   convert   - Convert old Office formats (.doc, .xls, .ppt)
echo   extract   - Extract text from all files
echo   embed     - Generate embeddings via cloud API
echo   all       - Run all steps in sequence
echo   web       - Start Streamlit web interface
echo   stats     - Show database statistics

:end
endlocal
