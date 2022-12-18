import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import get_apidata
import sys
import plotly.express as px
from arima import ArimaModel
from io import StringIO
import ta


st.set_page_config(page_title='BlockChain Project',page_icon="❄️",layout="wide")

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)


# Title
navbar = """
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #343434;">
  <p style="margin-bottom: 1px; margin-top: 3px; color:#FFFFFF; text-shadow: 2px 2px 4px #000000; font-size: 30px;">Crypto Price Prediction</p>
  <p style="margin-left: 6px; margin-bottom: 0px; margin-top: 12px; color:#FFFFFF; text-shadow: 1px 1px 2px #d3d3d3; font-size: 15px;">using ARIMA model</p>
  </button
</nav>
"""
st.markdown(navbar, unsafe_allow_html=True)

st.write("---")

tuple_currencies, coinname = get_apidata.getListCoins()

def main():
    # Initialize columns and select boxes
    col1, col2 = st.columns(2)
    with col2:
      date_period = st.selectbox("Choose the prediction period",("1 DAY", "1 WEEK", "2 WEEKS", "1 MONTH"))

    # Get and display data
    with col1:
        coins = st.selectbox("Which coin would you like to predict?", tuple_currencies)
        with st.expander("Data"):
            name = "Coin: " + coinname.get(coins)
            st.subheader(name)
            data = get_apidata.getFinalData(coins, date_period)
            st.dataframe(data)

    # Get and display graph
    with col2:
        with st.expander("Graph"):
            data["MA20"] = ta.trend.sma_indicator(data['close'], window=20)
            data["MA50"] = ta.trend.sma_indicator(data['close'], window=50)
            data["MA100"] = ta.trend.sma_indicator(data['close'], window=100)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_width=[0.2, 0.7])

            fig.add_trace(go.Candlestick(x=data.index,
                                  open=data['open'],
                                  high=data['high'],
                                  low=data['low'],
                                  close=data['close'], name="OHLC"),
                  row=1, col=1)
            fig.add_trace(go.Line(x=data.index, y=data['MA20'], name="MA20", line=dict(
            color="purple",
            width=1)))
            fig.add_trace(go.Line(x=data.index, y=data['MA50'], name="MA50", line=dict(
            color="yellow",
            width=1.5)))
            fig.add_trace(go.Line(x=data.index, y=data['MA100'], name="MA100", line=dict(
            color="orange",
            width=2)))

            # Bar trace for volumes on 2nd row without legend
            fig.add_trace(go.Bar(x=data.index, y=data['volume'], showlegend=False), row=2, col=1)

            fig.update(layout_xaxis_rangeslider_visible=False)

            fig.update_layout(autosize=False,width=780, height=540, margin=dict(l=50,r=50,b=50,t=50,pad=4))
            st.plotly_chart(fig)

    model = ArimaModel(data, date_period)
    
    st.write('---')    
    col1, col2 = st.columns(2)

    with col1:
        period = st.slider("Choose additional period for the prediction", 1, 10, 1)
        if date_period == '1 DAY':
          out = 'DAYS'
        if date_period == '1 WEEK':
          if period == 1:
            out = 'WEEK'
          else:
            out = 'WEEKS'
        if date_period == '2 WEEKS':
          if period == 1:
            out = 'WEEK'
          else:
            out = 'WEEKS'
        if date_period == '1 MONTH':
          if period == 1:
            out = 'MONTH'
          else:
            out = 'MONTHS'
        new_data = "+"+str(period)+" "+out
        st.write(new_data)
    with col2:
        st.markdown(' ')

 
    st.write("Press the **START** button to show the model and forecast results.")
    
    if st.button("START"):
      st.warning(model.checkData())
      model.createDataReturn()
      st.write("Stationality test")
      warn, ADF, p_value = model.checkStationarity()
      s1 = "ADF Statistic: " + str(ADF)
      s2 = "p-value: " + str(p_value)
      st.text(s1)
      st.text(s2)
      st.warning(warn)

      st.markdown("**Please wait...**")
      with st.expander("Summary SARIMAX Results"):
          result = model.displaySummary()

          old_stdout = sys.stdout
          sys.stdout = mystdout = StringIO()
          print(result.summary())
          sys.stdout = old_stdout
          st.text(mystdout.getvalue())

          pre = model.predict(period)
      
      col1, col2 = st.columns(2)

      with col1:
        st.write("Results of prediction:")
        st.dataframe(pre)

      with col2:
        fig_plot = px.line(data, y="close", x=data.index)
        fig_plot.add_trace(
            go.Scatter(x=pre.index, y=pre['Mean_Price'], line=dict(color="red"), name="forecast"))
        fig_plot.add_trace(go.Scatter(x=pre.index, y=pre['Upper_Price'], line=dict(color="green", dash='dash'), name="upper", ))
        fig_plot.add_trace(go.Scatter(x=pre.index, y=pre['Lower_Price'], line=dict(color="green", dash='dash'), name="lower", ))
        st.plotly_chart(fig_plot)
        
        st.markdown('---')
if __name__ == '__main__':
    main()

title = """
<p style="text-align: center; margin-top: 55px; color:#BEBEBE; text-shadow: 1px 1px 0px #000000; font-size: 13px;">Built by Joy Bugalia</p>
"""
st.markdown(title, unsafe_allow_html=True)