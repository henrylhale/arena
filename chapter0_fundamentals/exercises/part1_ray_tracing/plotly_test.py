import plotly.graph_objects as go

  fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 2], mode="lines+markers"))
  fig.update_layout(title="molten + plotly test")
  fig.show(renderer="browser")




