import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

def figText(ax,x,y,s,bgcolor='white',**kwargs):
	
	txt = ax.text(x,y,s,**kwargs)
	
	if not bgcolor is None:
		txt.set_path_effects([path_effects.Stroke(linewidth=2,foreground=bgcolor),path_effects.Normal()])

	return txt
