//+------------------------------------------------------------------+
//|                                           ExportIndicatorCSV.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Arrays\ArrayString.mqh>
#include <Indicators\Trend.mqh>
#include <Indicators\Oscilators.mqh>
#include <Indicators\Volumes.mqh>

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart() {
    main();
}
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|1 buffer apenas                                                   |
//+------------------------------------------------------------------+
void save_csv_1b(string file_name, MqlRates& rates[], double& buffer[]) {
    int fileHandle = FileOpen(file_name, FILE_WRITE|FILE_ANSI|FILE_CSV, '\t');
    Print(file_name);

    if(fileHandle==INVALID_HANDLE) {
        Alert("Error opening file");
        return;
    }

    string header = "DATETIME\tB0";
    FileWrite(fileHandle, header);

    for(int i=0; i < ArraySize(rates); i++) {
        string date_time = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES);
        StringReplace(date_time, ".", "-");
        StringReplace(date_time, " ", "T");
        FileWrite(fileHandle, date_time, buffer[i]);
    }

    FileClose(fileHandle);
}
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void save_csv_2b(string file_name, MqlRates& rates[], double& buffer_0[], double& buffer_1[]) {
    int fileHandle = FileOpen(file_name, FILE_WRITE|FILE_ANSI|FILE_CSV, '\t');
    Print(file_name);

    if(fileHandle==INVALID_HANDLE) {
        Alert("Error opening file");
        return;
    }

    string header = "DATETIME\tB0\tB1";
    FileWrite(fileHandle, header);

    for(int i=0; i < ArraySize(rates); i++) {
        string date_time = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES);
        StringReplace(date_time, ".", "-");
        StringReplace(date_time, " ", "T");
        FileWrite(fileHandle, date_time, buffer_0[i], buffer_1[i]);
    }

    FileClose(fileHandle);
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// this function exports the MA indicator to a CSV file.
// It returns the number of copied rates.
int Export_MA(string symbol_name, ENUM_TIMEFRAMES timeframe, int ma_period,
              datetime start_time, datetime stop_time) {
    int               ma_shift = 0;
    ENUM_MA_METHOD    ma_method = MODE_SMA;
    int               ma_applied = PRICE_CLOSE;
    CiMA ma = CiMA();
    MqlRates rates[];

    bool r = ma.Create(symbol_name, timeframe, ma_period, ma_shift, ma_method, ma_applied);
    if (!r) {
        Print("Erro ao criar o indicador.");
        return -1;
    }

    ma.Refresh();

    int n_copied_rates = CopyRates(symbol_name, timeframe, start_time, stop_time, rates);
    if(n_copied_rates <= 0) {
        Print("Error copying rates ",GetLastError());
        return -1;
    }
    Print("n_copied_rates = ", n_copied_rates);

    double buffer[];
    int n_ind_elems = ma.GetData(start_time, stop_time, 0, buffer);
    if (n_ind_elems == -1) {
        Print("Erro em GetData.");
        return -1;
    }

    if (n_copied_rates != n_ind_elems) {
        Print("Erro. n_copied_rates != n_ind_elems.");
        return -1;
    }

    string filename = symbol_name + "-MA(" + IntegerToString(ma_period) +
    ")_" + EnumToString(timeframe) + ".csv";
    StringReplace(filename, "PERIOD_", "");
    save_csv_1b(filename, rates, buffer);
    return n_copied_rates;
}
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// this function exports the Stochastic indicator to a CSV file.
// It returns the number of copied rates.
int Export_Stochastic(string symbol_name, ENUM_TIMEFRAMES timeframe, int k_period,
                      int d_period, int slowing,
                      datetime start_time, datetime stop_time) {
    ENUM_MA_METHOD    ma_method = MODE_SMA;
    ENUM_STO_PRICE price_field = STO_LOWHIGH;
    CiStochastic sto = CiStochastic();
    MqlRates rates[];

    bool r = sto.Create(symbol_name, timeframe, k_period, d_period, slowing, ma_method, price_field);
    if (!r) {
        Print("Erro ao criar o indicador.");
        return -1;
    }

    sto.Refresh();

    int n_copied_rates = CopyRates(symbol_name, timeframe, start_time, stop_time, rates);
    if(n_copied_rates <= 0) {
        Print("Error copying rates ",GetLastError());
        return -1;
    }
    Print("n_copied_rates = ", n_copied_rates);

    double buffer_0[];
    double buffer_1[];

    int n_ind_elems_0 = sto.GetData(start_time, stop_time, 0, buffer_0);
    if (n_ind_elems_0 == -1) {
        Print("Erro em GetData.");
        return -1;
    }

    if (n_copied_rates != n_ind_elems_0) {
        Print("Erro. n_copied_rates != n_ind_elems_0.");
        return -1;
    }

    int n_ind_elems_1 = sto.GetData(start_time, stop_time, 1, buffer_1);
    if (n_ind_elems_1 == -1) {
        Print("Erro em GetData.");
        return -1;
    }

    if (n_copied_rates != n_ind_elems_1) {
        Print("Erro. n_copied_rates != n_ind_elems_1.");
        return -1;
    }

    string filename = symbol_name + "-STO(" + IntegerToString(k_period) + "," +
                      IntegerToString(d_period) + "," + IntegerToString(slowing) + ")_" +
                      EnumToString(timeframe) + ".csv";
    StringReplace(filename, "PERIOD_", "");
    save_csv_2b(filename, rates, buffer_0, buffer_1);
    return n_copied_rates;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// this function exports the MACD indicator to a CSV file.
// It returns the number of copied rates.
int Export_MACD(string symbol_name, ENUM_TIMEFRAMES timeframe,
                int fast_ema_period, int slow_ema_period, int signal_period,
                datetime start_time, datetime stop_time) {
    ENUM_MA_METHOD    ma_method = MODE_SMA;
    ENUM_STO_PRICE price_field = STO_LOWHIGH;
    int applied = PRICE_CLOSE;
    CiMACD macd = CiMACD();
    MqlRates rates[];

    bool r = macd.Create(symbol_name, timeframe, fast_ema_period, slow_ema_period, signal_period, applied);
    if (!r) {
        Print("Erro ao criar o indicador.");
        return -1;
    }

    macd.Refresh();

    int n_copied_rates = CopyRates(symbol_name, timeframe, start_time, stop_time, rates);
    if(n_copied_rates <= 0) {
        Print("Error copying rates ",GetLastError());
        return -1;
    }
    Print("n_copied_rates = ", n_copied_rates);

    double buffer_0[];
    double buffer_1[];

    int n_ind_elems_0 = macd.GetData(start_time, stop_time, 0, buffer_0);
    if (n_ind_elems_0 == -1) {
        Print("Erro em GetData.");
        return -1;
    }

    if (n_copied_rates != n_ind_elems_0) {
        Print("Erro. n_copied_rates != n_ind_elems_0.");
        return -1;
    }

    int n_ind_elems_1 = macd.GetData(start_time, stop_time, 1, buffer_1);
    if (n_ind_elems_1 == -1) {
        Print("Erro em GetData.");
        return -1;
    }

    if (n_copied_rates != n_ind_elems_1) {
        Print("Erro. n_copied_rates != n_ind_elems_1.");
        return -1;
    }

    string filename = symbol_name + "-MACD(" + IntegerToString(fast_ema_period) + "," +
                      IntegerToString(slow_ema_period) + "," + IntegerToString(signal_period) +
                      ")_" + EnumToString(timeframe) + ".csv";
    StringReplace(filename, "PERIOD_", "");
    save_csv_2b(filename, rates, buffer_0, buffer_1);
    return n_copied_rates;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// this function exports the RSI indicator to a CSV file.
// It returns the number of copied rates.
int Export_RSI(string symbol_name, ENUM_TIMEFRAMES timeframe, int ma_period,
               datetime start_time, datetime stop_time) {
    int               ma_applied = PRICE_CLOSE;
    CiRSI rsi = CiRSI();
    MqlRates rates[];

    bool r = rsi.Create(symbol_name, timeframe, ma_period, ma_applied);
    if (!r) {
        Print("Erro ao criar o indicador.");
        return -1;
    }

    rsi.Refresh();

    int n_copied_rates = CopyRates(symbol_name, timeframe, start_time, stop_time, rates);
    if(n_copied_rates <= 0) {
        Print("Error copying rates ",GetLastError());
        return -1;
    }
    Print("n_copied_rates = ", n_copied_rates);

    double buffer[];
    int n_ind_elems = rsi.GetData(start_time, stop_time, 0, buffer);
    if (n_ind_elems == -1) {
        Print("Erro em GetData.");
        return -1;
    }

    if (n_copied_rates != n_ind_elems) {
        Print("Erro. n_copied_rates != n_ind_elems.");
        return -1;
    }

    string filename = symbol_name + "-RSI(" + IntegerToString(ma_period) + ")_" + EnumToString(timeframe) + ".csv";
    StringReplace(filename, "PERIOD_", "");
    save_csv_1b(filename, rates, buffer);
    return n_copied_rates;
}

// this function exports the OBV indicator to a CSV file.
// It returns the number of copied rates.
int Export_OBV(string symbol_name, ENUM_TIMEFRAMES timeframe,
              datetime start_time, datetime stop_time) {
    ENUM_APPLIED_VOLUME applied = VOLUME_TICK;
    CiOBV obv = CiOBV();
    MqlRates rates[];

    bool r = obv.Create(symbol_name, timeframe, applied);
    if (!r) {
        Print("Erro ao criar o indicador.");
        return -1;
    }

    obv.Refresh();

    int n_copied_rates = CopyRates(symbol_name, timeframe, start_time, stop_time, rates);
    if(n_copied_rates <= 0) {
        Print("Error copying rates ",GetLastError());
        return -1;
    }
    Print("n_copied_rates = ", n_copied_rates);

    double buffer[];
    int n_ind_elems = obv.GetData(start_time, stop_time, 0, buffer);
    if (n_ind_elems == -1) {
        Print("Erro em GetData.");
        return -1;
    }

    if (n_copied_rates != n_ind_elems) {
        Print("Erro. n_copied_rates != n_ind_elems.");
        return -1;
    }

    string filename = symbol_name + "-OBV" + "_" + EnumToString(timeframe) + ".csv";
    StringReplace(filename, "PERIOD_", "");
    save_csv_1b(filename, rates, buffer);
    return n_copied_rates;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// this function exports the indicators of a symbol to CSV files.
int Export_Symbol_Indicators(string symbol, ENUM_TIMEFRAMES timeframe, datetime start_time, datetime stop_time) {
    string            _symbol = symbol;
    ENUM_TIMEFRAMES   _timeframe = timeframe;
    /*datetime start_time = D'2018.01.02 02:00:00';
    datetime stop_time = D'2024.06.15 00:00:00'; XAUUSD*/
    datetime _start_time = start_time;
    datetime _stop_time = stop_time;

    // this function will return the total number of copied rates
    int total_copied_rates = 0;

    // MA
    total_copied_rates += Export_MA(_symbol, _timeframe, 1, _start_time, _stop_time);
    /*total_copied_rates += Export_MA(_symbol, _timeframe, 5, _start_time, _stop_time);
    total_copied_rates += Export_MA(_symbol, _timeframe, 10, _start_time, _stop_time);
    total_copied_rates += Export_MA(_symbol, _timeframe, 25, _start_time, _stop_time);
    total_copied_rates += Export_MA(_symbol, _timeframe, 50, _start_time, _stop_time);
    total_copied_rates += Export_MA(_symbol, _timeframe, 100, _start_time, _stop_time);
    total_copied_rates += Export_MA(_symbol, _timeframe, 200, _start_time, _stop_time);*/

    // Stochastic
    //total_copied_rates += Export_Stochastic(_symbol, _timeframe, 5, 3, 3, _start_time, _stop_time);

    // MACD
    //total_copied_rates += Export_MACD(_symbol, _timeframe, 13, 26, 9, _start_time, _stop_time);

    // RSI
    //total_copied_rates += Export_RSI(_symbol, _timeframe, 13, _start_time, _stop_time);

    // OBV
    //total_copied_rates += Export_OBV(_symbol, _timeframe, _start_time, _stop_time);

    Print("total_copied_rates = ", total_copied_rates);

    return total_copied_rates;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void main() {
    ENUM_TIMEFRAMES   timeframe = PERIOD_M15;
    datetime start_time = D'2018.01.02 02:00:00';
    datetime stop_time = D'2024.06.15 00:00:00';
    
    // forex majors symbols
    string majors_symbols[] = {
        "AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"
    };

    CArrayString cas_symbols;
    cas_symbols.AddArray(majors_symbols);

    int num_symbols = cas_symbols.Total();
    Print("num_symbols = ", num_symbols);

    cas_symbols.Sort();

    int total_copied_rates = 0;

    for(int i=0; i < num_symbols; i++) {
        string symbol = cas_symbols.At(i);
        Print("symbol = ", symbol);
        //Export_Symbol_Indicators(symbol, timeframe, start_time, stop_time);
        total_copied_rates += Export_Symbol_Indicators(symbol, timeframe, start_time, stop_time);
    }

    Print("total_copied_rates = ", total_copied_rates);
}
//+------------------------------------------------------------------+
