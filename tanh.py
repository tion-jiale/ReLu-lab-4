import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Visualizer", layout="wide")
st.title("Tanh Activation Function")

st.markdown("""
**Tanh (Hyperbolic Tangent)** outputs values between -1 and 1 and is zero-centered.
""")

# Sidebar
st.sidebar.header("Graph Settings")
x_min = st.sidebar.slider("Min X value", -10.0, -1.0, -10.0)
x_max = st.sidebar.slider("Max X value", 1.0, 10.0, 10.0)

x = np.linspace(x_min, x_max, 100)

def tanh(x):
    return np.tanh(x)

y = tanh(x)

test_val = st.number_input("Enter a test value for x:", value=0.0)
res_val = tanh(test_val)

st.latex(r"f(x) = \tanh(x)")
st.write(f"**Result f({test_val}) = {res_val:.4f}**")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, linewidth=2)
ax.axhline(0, linestyle="--")
ax.axvline(0, linestyle="--")
ax.grid(True)

if x_min <= test_val <= x_max:
    ax.plot(test_val, res_val, "ro")

ax.set_title("Tanh Function")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")

st.pyplot(fig)
