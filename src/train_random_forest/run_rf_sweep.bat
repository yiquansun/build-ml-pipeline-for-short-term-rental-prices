REM @echo off
REM SETLOCAL EnableDelayedExpansion

REM ========================================
REM Random Forest Sweep BAT File
REM ========================================

REM SET MAX_TFIDF=10
REM SET MAX_FEATURES_LIST=0.1 0.33 0.5 0.75 1.0
REM SET STRATIFY_COL=neighbourhood_group
REM SET TRAINVAL_ARTIFACT=yiquan_sun-cariad/nyc_airbnb/trainval_data.csv:v0
REM SET RF_CONFIG=rf_config.json

REM FOR %%M IN (%MAX_FEATURES_LIST%) DO (
    
    REM SET "MF=%%M"
    REM SET "OUTPUT_NAME=rf_tfidf%MAX_TFIDF%_mf!MF!"
    
    REM ECHO ======================================================
    REM ECHO Running combination Max_Features=!MF!
    REM ECHO ======================================================

    REM Update rf_config.json
    REM We use !MF! here because we are inside a loop with delayed expansion
    REM python -c "import json; f=open('%RF_CONFIG%','r'); c=json.load(f); c['max_features']=!MF!; f.close(); f=open('%RF_CONFIG%','w'); json.dump(c,f); f.close()"

    REM Run training
    REM python run.py ^
        REM --trainval_artifact "%TRAINVAL_ARTIFACT%" ^
        REM --val_size 0.2 ^
        REM --random_seed 42 ^
        REM --stratify_by "%STRATIFY_COL%" ^
        REM --rf_config "%RF_CONFIG%" ^
        REM --max_tfidf_features %MAX_TFIDF% ^
        REM --output_artifact "!OUTPUT_NAME!"
REM )
REM PAUSE


@echo off
SETLOCAL EnableDelayedExpansion

SET MAX_TFIDF=10
SET MAX_FEATURES_LIST=0.1 0.33 0.5 0.75 1.0
SET STRATIFY_COL=neighbourhood_group
SET TRAINVAL_ARTIFACT=yiquan_sun-cariad/nyc_airbnb/trainval_data.csv:v0
SET RF_CONFIG=rf_config.json

FOR %%M IN (%MAX_FEATURES_LIST%) DO (
    
    SET "MF=%%M"
    SET "OUTPUT_NAME=rf_tfidf%MAX_TFIDF%_mf!MF!"
    
    ECHO ======================================================
    ECHO Running combination Max_Features=!MF!
    ECHO ======================================================

    REM This Python logic loads the existing JSON and updates ONLY max_features
    python -c "import json; f=open('%RF_CONFIG%','r'); c=json.load(f); c['max_features']=!MF!; f.close(); f=open('%RF_CONFIG%','w'); json.dump(c,f,indent=4); f.close()"

    python run.py ^
        --trainval_artifact "%TRAINVAL_ARTIFACT%" ^
        --val_size 0.2 ^
        --random_seed 42 ^
        --stratify_by "%STRATIFY_COL%" ^
        --rf_config "%RF_CONFIG%" ^
        --max_tfidf_features %MAX_TFIDF% ^
        --output_artifact "!OUTPUT_NAME!"
)
PAUSE