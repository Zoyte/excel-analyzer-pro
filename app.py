from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import json
import os
import io
import warnings
warnings.filterwarnings('ignore')

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'excel_analyzer_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

dataframes = {}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def convert_df_to_json(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if hasattr(df_copy[col], 'dt'):
            try:
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d')
            except Exception:
                pass
    return df_copy.where(pd.notnull(df_copy), None).to_dict(orient='records')


def get_column_stats(df, col):
    stats = {}
    if df[col].dtype in ['int64', 'float64']:
        stats['type'] = 'numeric'
        stats['min']    = float(df[col].min())    if not pd.isna(df[col].min())    else None
        stats['max']    = float(df[col].max())    if not pd.isna(df[col].max())    else None
        stats['mean']   = float(df[col].mean())   if not pd.isna(df[col].mean())   else None
        stats['median'] = float(df[col].median()) if not pd.isna(df[col].median()) else None
        stats['std']    = float(df[col].std())    if not pd.isna(df[col].std())    else None
        stats['sum']    = float(df[col].sum())    if not pd.isna(df[col].sum())    else None
        stats['null_count']   = int(df[col].isna().sum())
        stats['unique_count'] = int(df[col].nunique())
    elif str(df[col].dtype).startswith('datetime'):
        stats['type'] = 'datetime'
        stats['min']  = str(df[col].min()) if not pd.isna(df[col].min()) else None
        stats['max']  = str(df[col].max()) if not pd.isna(df[col].max()) else None
        stats['null_count']   = int(df[col].isna().sum())
        stats['unique_count'] = int(df[col].nunique())
    else:
        stats['type'] = 'categorical'
        stats['unique_count'] = int(df[col].nunique())
        stats['null_count']   = int(df[col].isna().sum())
        top = df[col].value_counts().head(10)
        stats['top_values'] = {str(k): int(v) for k, v in top.items()}
    return stats


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не найден'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[-1].lower()
        if ext not in ['xlsx', 'xls', 'csv']:
            return jsonify({'error': 'Поддерживаются только .xlsx, .xls, .csv'}), 400

        file_bytes = file.read()
        df = pd.read_excel(io.BytesIO(file_bytes)) if ext in ['xlsx', 'xls'] else \
             pd.read_csv(io.BytesIO(file_bytes))

        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

        dataframes['user_df'] = df

        columns_info = []
        for col in df.columns:
            ct = str(df[col].dtype)
            kind = 'numeric' if ('int' in ct or 'float' in ct) else \
                   'datetime' if 'datetime' in ct else 'categorical'
            columns_info.append({
                'name': col, 'dtype': ct, 'kind': kind,
                'null_count': int(df[col].isna().sum()),
                'unique_count': int(df[col].nunique())
            })

        return jsonify({
            'success': True, 'filename': filename,
            'rows': len(df), 'columns': len(df.columns),
            'columns_info': columns_info,
            'preview': convert_df_to_json(df.head(50))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/filter', methods=['POST'])
def apply_filter():
    try:
        if 'user_df' not in dataframes:
            return jsonify({'error': 'Данные не загружены'}), 400

        df   = dataframes['user_df'].copy()
        data = request.get_json()
        filters  = data.get('filters', [])
        sort_by  = data.get('sort_by', None)
        sort_asc = data.get('sort_asc', True)
        group_by = data.get('group_by', None)

        # ── apply filters ──────────────────────────────
        for f in filters:
            col  = f.get('column')
            op   = f.get('operator')
            val  = f.get('value')
            val2 = f.get('value2')
            if col not in df.columns:
                continue
            try:
                if   op == 'equals':      df = df[df[col].astype(str).str.lower() == str(val).lower()] if df[col].dtype == object else df[df[col] == float(val)]
                elif op == 'not_equals':  df = df[df[col].astype(str).str.lower() != str(val).lower()] if df[col].dtype == object else df[df[col] != float(val)]
                elif op == 'contains':    df = df[df[col].astype(str).str.lower().str.contains(str(val).lower(), na=False)]
                elif op == 'not_contains':df = df[~df[col].astype(str).str.lower().str.contains(str(val).lower(), na=False)]
                elif op == 'greater':     df = df[df[col] > float(val)]
                elif op == 'less':        df = df[df[col] < float(val)]
                elif op == 'greater_eq':  df = df[df[col] >= float(val)]
                elif op == 'less_eq':     df = df[df[col] <= float(val)]
                elif op == 'between':     df = df[(df[col] >= float(val)) & (df[col] <= float(val2))]
                elif op == 'in_list':     df = df[df[col].astype(str).isin([v.strip() for v in str(val).split(',')])]
                elif op == 'is_null':     df = df[df[col].isna()]
                elif op == 'not_null':    df = df[df[col].notna()]
                elif op == 'date_from':   df = df[df[col] >= pd.to_datetime(val)]
                elif op == 'date_to':     df = df[df[col] <= pd.to_datetime(val)]
                elif op == 'month_in':
                    months = [int(m.strip()) for m in str(val).split(',') if m.strip().isdigit()]
                    if months:
                        df = df[pd.to_datetime(df[col], errors='coerce').dt.month.isin(months)]
                # ── price-specific ──
                elif op == 'pct_change_gt':
                    pct = df[col].pct_change() * 100
                    df = df[pct > float(val)]
                elif op == 'pct_change_lt':
                    pct = df[col].pct_change() * 100
                    df = df[pct < float(val)]
                elif op == 'trend_up':
                    df = df[df[col].pct_change() > 0]
                elif op == 'trend_down':
                    df = df[df[col].pct_change() < 0]
                elif op == 'top_n':
                    df = df.nlargest(int(float(val)), col)
                elif op == 'bottom_n':
                    df = df.nsmallest(int(float(val)), col)
                elif op == 'above_avg':
                    df = df[df[col] > df[col].mean()]
                elif op == 'below_avg':
                    df = df[df[col] < df[col].mean()]
            except Exception:
                continue

        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=sort_asc)

        group_result = None
        if group_by and group_by in df.columns:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                grp = df.groupby(group_by)[num_cols].agg(['sum', 'mean', 'count']).round(2)
                grp.columns = ['_'.join(c) for c in grp.columns]
                grp = grp.reset_index()
                group_result = convert_df_to_json(grp)

        stats = {col: get_column_stats(df, col) for col in df.columns}

        # Save filtered df so forecast can use it
        dataframes['filtered_df'] = df

        return jsonify({
            'success': True, 'rows': len(df),
            'preview': convert_df_to_json(df.head(200)),
            'stats': stats, 'group_result': group_result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        if 'user_df' not in dataframes:
            return jsonify({'error': 'Данные не загружены'}), 400
        df = dataframes.get('filtered_df', dataframes['user_df'])
        stats = {col: get_column_stats(df, col) for col in df.columns}
        return jsonify({'success': True, 'stats': stats, 'rows': len(df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# DEBUG: see what's in the dataframe
# ─────────────────────────────────────────────
@app.route('/debug_cols', methods=['GET'])
def debug_cols():
    if 'user_df' not in dataframes:
        return jsonify({'error': 'no data'}), 400
    df = dataframes['user_df']
    info = []
    for col in df.columns:
        sample = df[col].dropna().head(3).tolist()
        info.append({'col': col, 'dtype': str(df[col].dtype), 'sample': [str(s) for s in sample]})
    return jsonify({'cols': info})


# ─────────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────────
@app.route('/forecast', methods=['POST'])
def forecast():
    import traceback
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline

        if 'user_df' not in dataframes:
            return jsonify({'error': 'Данные не загружены — загрузи Excel файл'}), 400

        data      = request.get_json(force=True)
        date_col  = (data.get('date_col') or '').strip()
        price_col = (data.get('price_col') or '').strip()
        freq      = data.get('freq', 'ME')
        periods   = int(data.get('periods', 3))

        # Upgrade legacy freq strings for pandas 2.2.0+
        if freq == 'M': freq = 'ME'
        if freq == 'Q': freq = 'QE'
        if freq == 'W': freq = 'W-MON'

        if not date_col:
            return jsonify({'error': 'Не выбрана колонка даты'}), 400
        if not price_col:
            return jsonify({'error': 'Не выбрана колонка цены'}), 400

        # Use filtered data if available, otherwise raw
        df = dataframes.get('filtered_df', dataframes['user_df']).copy()

        if date_col not in df.columns:
            return jsonify({'error': f'Колонка даты "{date_col}" не найдена. Доступные: {list(df.columns)}'}), 400
        if price_col not in df.columns:
            return jsonify({'error': f'Колонка цены "{price_col}" не найдена'}), 400

        # ── Parse dates (try several strategies) ──
        date_series = df[date_col]
        parsed = None
        for kwargs in [
            {},
            {'dayfirst': True},
            {'format': 'mixed'},
            {'format': '%d.%m.%Y'},
            {'format': '%Y-%m-%d'},
            {'format': '%m/%d/%Y'},
            {'format': '%d/%m/%Y'},
        ]:
            try:
                parsed = pd.to_datetime(date_series, errors='raise', **kwargs)
                if parsed.isna().all():
                    parsed = None
                    continue
                break
            except Exception:
                continue

        if parsed is None:
            sample = df[date_col].dropna().head(5).tolist()
            return jsonify({
                'error': f'Не удалось распознать даты в колонке "{date_col}". '
                         f'Примеры значений в колонке: {sample}. '
                         f'Поддерживаемые форматы: 2024-01-15, 15.01.2024, 01/15/2024'
            }), 400

        df[date_col] = parsed

        # ── Parse price ──
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

        df_clean = df[[date_col, price_col]].dropna()
        if len(df_clean) == 0:
            return jsonify({'error': f'После очистки данных не осталось строк. Проверь что в колонке "{price_col}" есть числа'}), 400

        df_clean = df_clean.sort_values(date_col)
        df_clean = df_clean.groupby(date_col)[price_col].mean().reset_index()
        df_clean = df_clean.set_index(date_col)

        # Resample
        try:
            df_resampled = df_clean.resample(freq).mean()
            df_resampled = df_resampled.interpolate(method='linear')
        except Exception as re:
            return jsonify({'error': f'Ошибка при ресемплинге по частоте "{freq}": {str(re)}'}), 400

        series = df_resampled[price_col].dropna()

        if len(series) < 4:
            return jsonify({
                'error': f'Недостаточно данных для прогноза: после агрегации по частоте "{freq}" получилось только {len(series)} точек. '
                         f'Нужно минимум 4. Попробуй выбрать другую частоту (например, "Ежедневно" вместо "Ежемесячно")'
            }), 400

        # ── Walk-forward validation ──
        n_val = max(1, min(3, len(series) // 4))
        if len(series) - n_val < 3:
            n_val = 1
        train  = series.iloc[:-n_val]
        actual = series.iloc[-n_val:]
        results = {}

        def mape(y_true, y_pred):
            y_true = np.array(y_true, dtype=float)
            y_pred = np.array(y_pred, dtype=float)
            mask = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
            if not mask.any():
                return 999.0
            return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

        # Model A: Simple Exponential Smoothing (always works)
        try:
            m = ExponentialSmoothing(train, trend='add', damped_trend=True).fit(optimized=True)
            p = m.forecast(n_val)
            results['ets'] = {'model': m, 'mape': mape(actual.values, p.values), 'label': 'Exp.Smoothing', 'kind': 'ets'}
        except Exception:
            pass

        # Model B: Holt-Winters with seasonal (only if enough data)
        if len(train) >= 16:
            for sp in [12, 4, 7]:
                if len(train) >= 2 * sp:
                    try:
                        m = ExponentialSmoothing(train, trend='add', seasonal='add',
                                                 seasonal_periods=sp, damped_trend=True).fit(optimized=True)
                        p = m.forecast(n_val)
                        results['hw'] = {'model': m, 'mape': mape(actual.values, p.values),
                                         'label': f'Holt-Winters(s={sp})', 'kind': 'ets'}
                        break
                    except Exception:
                        continue

        # Model C: ARIMA
        best_arima = None
        best_arima_mape = 999
        best_order = (1, 1, 0)
        for p_val in [1, 2]:
            for q_val in [0, 1]:
                try:
                    m = ARIMA(train, order=(p_val, 1, q_val)).fit()
                    p = m.forecast(n_val)
                    err = mape(actual.values, np.array(p))
                    if err < best_arima_mape:
                        best_arima_mape = err
                        best_arima = m
                        best_order = (p_val, 1, q_val)
                except Exception:
                    continue
        if best_arima is not None:
            results['arima'] = {'model': best_arima, 'mape': best_arima_mape,
                                'label': f'ARIMA{best_order}', 'kind': 'arima', 'order': best_order}

        # Model D: Polynomial trend (always works)
        try:
            x = np.arange(len(train)).reshape(-1, 1)
            y = train.values.astype(float)
            pipe = Pipeline([('poly', PolynomialFeatures(degree=min(2, len(train)-1))),
                             ('ridge', Ridge(alpha=1.0))])
            pipe.fit(x, y)
            x_val = np.arange(len(train), len(train) + n_val).reshape(-1, 1)
            pred_poly = pipe.predict(x_val)
            results['poly'] = {'model': pipe, 'mape': mape(actual.values, pred_poly),
                               'label': 'Polynomial Trend', 'kind': 'sklearn', 'train_len': len(train)}
        except Exception:
            pass

        # Model E: Naive trend (absolute fallback)
        try:
            last = float(series.iloc[-1])
            slope = float(series.diff().dropna().mean())
            naive_pred = np.array([last + slope * (i+1) for i in range(n_val)])
            results['naive'] = {'mape': mape(actual.values, naive_pred),
                                'label': 'Naive Trend', 'kind': 'naive',
                                'last': last, 'slope': slope}
        except Exception:
            pass

        if not results:
            return jsonify({'error': 'Не удалось построить ни одну модель. Данных слишком мало или они некорректны'}), 500

        # ── Select best ──
        best_key = min(results, key=lambda k: results[k]['mape'])
        best     = results[best_key]

        # ── Forecast future ──
        future_vals  = []
        future_ci_lo = []
        future_ci_hi = []

        kind = best.get('kind', '')

        if kind == 'sklearn':
            pipe      = best['model']
            tl        = best['train_len']
            x_full    = np.arange(len(series) + periods).reshape(-1, 1)
            all_pred  = pipe.predict(x_full)
            resid_std = float(np.std(series.values - pipe.predict(np.arange(len(series)).reshape(-1, 1))))
            future_vals  = all_pred[-periods:].tolist()
            future_ci_lo = [v - 2*resid_std for v in future_vals]
            future_ci_hi = [v + 2*resid_std for v in future_vals]

        elif kind == 'ets':
            try:
                m2 = best['model']
                # refit on full series
                init_params = {'trend': m2.model.trend, 'damped_trend': m2.model.damped_trend}
                if hasattr(m2.model, 'seasonal') and m2.model.seasonal:
                    init_params['seasonal'] = m2.model.seasonal
                    init_params['seasonal_periods'] = m2.model.seasonal_periods
                m_full = ExponentialSmoothing(series, **init_params).fit(optimized=True)
                fc = m_full.forecast(periods)
                future_vals = fc.tolist()
                resid_std = float(np.std(m_full.resid))
                future_ci_lo = [v - 2*resid_std for v in future_vals]
                future_ci_hi = [v + 2*resid_std for v in future_vals]
            except Exception:
                kind = 'naive'  # fallback

        if kind == 'arima':
            try:
                order = best.get('order', (1, 1, 0))
                m_full = ARIMA(series, order=order).fit()
                fc_res = m_full.get_forecast(periods)
                future_vals  = fc_res.predicted_mean.tolist()
                ci = fc_res.conf_int(alpha=0.05)
                future_ci_lo = ci.iloc[:, 0].tolist()
                future_ci_hi = ci.iloc[:, 1].tolist()
            except Exception:
                kind = 'naive'

        if kind == 'naive' or not future_vals:
            last  = float(series.iloc[-1])
            slope = float(series.diff().dropna().mean())
            future_vals  = [last + slope * (i+1) for i in range(periods)]
            future_ci_lo = [v * 0.98 for v in future_vals]
            future_ci_hi = [v * 1.02 for v in future_vals]

        # ── Future dates ──
        last_date = series.index[-1]
        try:
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        except Exception:
            future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='ME')[1:]
        future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]

        hist_dates  = [d.strftime('%Y-%m-%d') for d in series.index]
        hist_values = [float(v) for v in series.values]

        last_price    = float(series.iloc[-1])
        fc_pct_change = [(v - last_price) / last_price * 100 if last_price != 0 else 0 for v in future_vals]

        vol_series = series.pct_change().dropna() * 100
        volatility = float(vol_series.std()) if len(vol_series) > 1 else 0.0

        model_scores = {v['label']: round(v['mape'], 4) for v in results.values()}

        return jsonify({
            'success'      : True,
            'best_model'   : best['label'],
            'mape'         : round(best['mape'], 4),
            'model_scores' : model_scores,
            'hist_dates'   : hist_dates,
            'hist_values'  : hist_values,
            'future_dates' : future_dates_str,
            'future_values': [round(float(v), 6) for v in future_vals],
            'ci_lower'     : [round(float(v), 6) for v in future_ci_lo],
            'ci_upper'     : [round(float(v), 6) for v in future_ci_hi],
            'last_price'   : round(last_price, 6),
            'fc_pct_change': [round(v, 4) for v in fc_pct_change],
            'volatility'   : round(volatility, 4),
            'n_hist_points': len(series),
            'freq'         : freq,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("FORECAST ERROR:", tb)
        return jsonify({'error': str(e), 'trace': tb}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

