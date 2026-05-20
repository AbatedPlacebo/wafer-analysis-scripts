# wafer-analysis-scripts
Before executing wafer\_analysis.py you should download dataset first.
https://drive.google.com/drive/folders/1g6J_3pcY9s09SbiTxVFQXECrbTiVHaFH?usp=drive_link

## Usage

usage: wafer_analysis.py [-h] --video VIDEO [--trial_id TRIAL_ID] --phase
                         {pre,post} [--odir ODIR] [--out_csv OUT_CSV]
                         [--out_video OUT_VIDEO] [--out_plot OUT_PLOT]
                         [--out_summary OUT_SUMMARY] [--out_json OUT_JSON]
                         [--um_per_px_x UM_PER_PX_X]
                         [--um_per_px_y UM_PER_PX_Y] [--side {left,right}]
                         [--roi X Y W H] [--select_roi] [--display]
                         [--start_frame START_FRAME] [--max_frames MAX_FRAMES]
                         [--smooth SMOOTH]
                         [--n_measurement_points N_MEASUREMENT_POINTS]
                         [--min_inliers MIN_INLIERS]
                         [--min_phase_response MIN_PHASE_RESPONSE]
                         [--trend_window TREND_WINDOW]
                         [--mad_thresh MAD_THRESH] [--flat_guard FLAT_GUARD]
                         [--max_frame_diff MAX_FRAME_DIFF]
                         [--required_stable_frames REQUIRED_STABLE_FRAMES]
                         [--min_search_x MIN_SEARCH_X]
