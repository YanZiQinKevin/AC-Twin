"""Compact qualitative architecture and per-seed scoring schematic."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Circle
from matplotlib.colors import to_rgb

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'mathtext.fontset':'dejavusans',
                     'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7.086614,4.527559))  # 180 x 115 mm
ax=fig.add_axes([.01,.01,.98,.98]); ax.set_xlim(0,20); ax.set_ylim(0,12); ax.axis('off')
ink='#253342'; blue='#4D93D6'; pale='#A4C8EA'; red='#D57873'

def text(x,y,s,size=7.5,bold=False,ha='center',color=ink):
    return ax.text(x,y,s,fontsize=max(7.5,size),fontweight='bold' if bold else 'normal',
                   ha=ha,va='center',color=color,zorder=10)

def box(x,y,w,h,label,face='white',edge=ink):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.12',
                              fc=face,ec=edge,lw=.7,zorder=5))
    text(x+w/2,y+h/2,label)

def group(x,y,w,h,letter,title,face):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.26',
                              fc=face,ec=ink,lw=.9,linestyle=(0,(5,4)),zorder=0))
    text(x+.25,y+h-.35,letter,9.6,True,ha='left')
    text(x+.7,y+h-.35,title,8.6,True,ha='left')

def block(x,y,w,h,color,label='',depth=.2):
    rgb=to_rgb(color)
    light=tuple(min(1,v+.12) for v in rgb); dark=tuple(v*.78 for v in rgb)
    ax.add_patch(Polygon([(x,y+h),(x+depth,y+h+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                         fc=light,ec='white',lw=.5,zorder=5))
    ax.add_patch(Polygon([(x+w,y),(x+w+depth,y+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                         fc=dark,ec='white',lw=.5,zorder=5))
    ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],fc=color,ec='white',lw=.5,zorder=6))
    if label: text(x+w/2,y+h/2,label,8)

def arrow(points,color=ink,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.9,
                linestyle=(0,(4,3)) if dashed else '-',zorder=3)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=8,
                                color=color,lw=.9,linestyle=(0,(4,3)) if dashed else '-',zorder=3))

group(.25,6.3,6.5,5.3,'a','Features','#FAFAFA')
group(7.05,6.3,12.7,5.3,'b','Direct audit head','#F5F9FD')
group(.25,.35,6.5,5.25,'c','Training only','#FFF8F4')
group(7.05,.35,12.7,5.25,'d','Risk score','#F5F9FD')

box(.65,8.5,2.18,1.5,'Predictions\n+ context','#D9D9D9')
text(1.74,7.92,'Frozen inputs')
block(3.5,8.7,1.45,1.4,blue)
for j in range(1,9):
    ax.plot([3.5,4.95],[8.7+j*1.4/9]*2,color='white',lw=.42,zorder=7)
for j in range(1,4):
    ax.plot([3.5+j*1.45/4]*2,[8.7,10.1],color='white',lw=.42,zorder=7)
text(4.23,10.68,r'$9\times34$',8,True)
text(4.23,8.21,'Tokens')
block(3.5,6.99,1.45,.65,pale,'9')
text(4.23,6.55,'Global')
block(5.77,8.57,.62,1.65,blue,'315')
arrow([(2.87,9.4),(3.49,9.4)])
arrow([(2.87,8.82),(3.15,8.82),(3.15,7.33),(3.49,7.33)])
arrow([(5.17,9.38),(5.75,9.38)])
arrow([(5.17,7.33),(5.44,7.33),(5.44,8.84),(5.75,8.84)])
text(6.1,7.98,'Concat')

block(7.67,8.52,1.05,1.72,pale,'43')
block(9.76,8.52,1.05,1.72,blue,'43')
block(11.86,8.74,.65,1.28,blue,'2')
arrow([(6.6,9.38),(7.65,9.38)])
arrow([(8.95,9.38),(9.74,9.38)])
arrow([(11.04,9.38),(11.84,9.38)])
text(9.35,7.69,'Linear → LayerNorm\nReLU → Dropout')
box(13.47,9.45,1.85,.76,'raw_mu','#DFECF8')
box(13.47,7.75,1.85,.76,'logit_high','#DFECF8')
arrow([(12.74,9.56),(13.1,9.56),(13.1,9.83),(13.45,9.83)])
arrow([(12.74,9.1),(13.1,9.1),(13.1,8.13),(13.45,8.13)])
arrow([(15.4,9.83),(18.12,9.83)])
arrow([(15.4,8.13),(18.12,8.13)])
text(16.84,10.36,'Softplus')
text(16.84,8.65,'Sigmoid')
text(18.6,9.83,r'$\mu^{+}$',9)
text(18.6,8.13,r'$p$',9)

box(.77,2.49,1.72,1.25,'True\nregret','#F1C2AD',red)
box(3.7,2.49,2.48,1.25,'Joint\nloss','#FBE7DD',red)
arrow([(2.55,3.11),(3.63,3.11)],red,True)
text(4.94,1.77,'SmoothL1\n+ 0.25 BCE',color=red)
arrow([(6.22,3.12),(6.9,3.12),(6.9,7.04),(7.6,7.04)],red,True)

# The two outputs of each seed are combined before averaging the five scores.
box(16.06,3.93,2.17,.85,'h · expm1','#DFECF8')
text(17.12,2.39,r'$p$',9)
ax.add_patch(Circle((14.68,3.13),.34,fc='white',ec=ink,lw=.8,zorder=5))
text(14.68,3.13,'×',10)
arrow([(18.6,9.56),(19.38,9.56),(19.38,4.36),(18.28,4.36)])
arrow([(18.6,7.86),(18.9,7.86),(18.9,2.4),(17.55,2.4)])
arrow([(16.02,4.36),(15.42,4.36),(15.42,3.29),(15.04,3.29)])
arrow([(16.72,2.4),(15.42,2.4),(15.42,2.99),(15.04,2.99)])
box(11.4,2.57,2.03,1.13,'Per-seed\nscore','#DFECF8')
box(7.85,2.57,2.42,1.13,'Mean over\n5 seeds','#A4C8EA')
arrow([(14.29,3.13),(13.49,3.13)])
arrow([(11.34,3.13),(10.33,3.13)])
text(12.42,1.91,r'$s_g^{(m)}$',9)
text(9.06,1.91,r'$\bar{s}_g$',9)
text(13.1,.92,'Higher score → higher review priority')

fig.savefig(ROOT/'Direct_audit_head_architecture_v2.svg',facecolor='white')
fig.savefig(ROOT/'Direct_audit_head_architecture_v2.pdf',facecolor='white')
fig.savefig(ROOT/'Direct_audit_head_architecture_v2.png',dpi=600,facecolor='white')
plt.close(fig)
print('Saved compact architecture and scoring figure')
