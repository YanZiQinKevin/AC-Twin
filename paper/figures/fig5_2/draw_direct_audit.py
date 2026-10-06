from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7,
                     'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'axes.linewidth': .8})
fig = plt.figure(figsize=(7.086614, 5.118110))  # 180 x 130 mm
ax = fig.add_axes([.01,.01,.98,.98])
ax.set_xlim(0,20); ax.set_ylim(0,13); ax.axis('off')
ink='#253342'; teal='#4D93D6'; blue='#A4C8EA'; purple='#4D93D6'; orange='#D57873'

def txt(x,y,s,size=7,weight='normal',color=ink,ha='center',va='center'):
    return ax.text(x,y,s,fontsize=max(size,7.5),fontweight=weight,color=color,ha=ha,va=va,zorder=9)

def box(x,y,w,h,label='',face='white',size=7,edge=ink):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.15',
                    fc=face,ec=edge,lw=.7,zorder=4)
    ax.add_patch(p)
    if label: txt(x+w/2,y+h/2,label,size)
    return p

def group(x,y,w,h,title,face):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=.32',
                    fc=face,ec=ink,lw=.95,linestyle=(0,(5,4)),zorder=0)
    ax.add_patch(p)
    letter, phrase = title.split('  ',1)
    txt(x+.27,y+h-.4,letter,9.6,'bold',ha='left')
    txt(x+.76,y+h-.4,phrase,8.6,'bold',ha='left')

def block(x,y,w,h,color,label='',size=7,depth=.25):
    rgb=to_rgb(color)
    top=tuple(min(1,c+.12) for c in rgb); side=tuple(c*.78 for c in rgb)
    ax.add_patch(Polygon([(x,y+h),(x+depth,y+h+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                         fc=top,ec='white',lw=.55,zorder=5))
    ax.add_patch(Polygon([(x+w,y),(x+w+depth,y+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
                         fc=side,ec='white',lw=.55,zorder=5))
    ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],fc=color,ec='white',lw=.55,zorder=6))
    if label: txt(x+w/2,y+h/2,label,size)

def arrow(points,dashed=False,color=ink,lw=1):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,
                linestyle=(0,(4,3)) if dashed else '-',zorder=3)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=8,
                               lw=lw,color=color,linestyle=(0,(4,3)) if dashed else '-',zorder=3))

group(.25,5.25,8.35,7.15,'a  Relative action features','#FAFAFA')
group(8.95,5.25,10.8,7.15,'b  Direct audit head','#F5F9FD')
group(.25,.4,19.5,4.2,'c  Joint training supervision','#FFF8F4')

box(.65,9.0,2.7,1.55,'Frozen predictions\n+ scene context',face='#D9D9D9',size=7.5)
txt(2,8.48,'Selected action '+r'$a_0$',7)
txt(2,8.02,'9 competing actions',7)
txt(2,7.18,'Costs and ranks\n5 risk channels',7.5)

block(4.05,8.75,2.12,1.65,teal,depth=.2)
for j in range(1,9):
    ax.plot([4.05,6.17],[8.75+j*1.65/9]*2,color='white',lw=.45,zorder=7)
for j in range(1,5):
    ax.plot([4.05+j*2.12/5]*2,[8.75,10.4],color='white',lw=.45,zorder=7)
txt(5.15,10.97,r'$9\times34$ tokens',8,'bold')
txt(5.15,8.05,'20 continuous\n14 one-hot',7.5)
block(4.05,6.6,2.12,.68,blue,'9 global features',6.6,depth=.2)
txt(4.9,5.92,'Gaps, agreement\nand scene context',7.5)
arrow([(3.37,9.85),(3.75,9.85),(4.04,9.85)])
arrow([(3.35,9.3),(3.62,9.3),(3.62,6.95),(4.04,6.95)])

block(7.25,8.15,.85,2.15,purple,'315',7.8,depth=.18)
arrow([(6.39,9.55),(7.23,9.55)])
arrow([(6.39,6.97),(6.86,6.97),(6.86,8.62),(7.23,8.62)])
txt(6.95,10.67,'Flatten',6.4)
txt(7.65,7.63,'Concat',7.5)
txt(7.65,7.09,r'$306+9$',7)
txt(7.65,6.32,'Train-fitted\nnormalization',6.2)

block(9.52,8.25,1.12,2.0,blue,'43',8.3,depth=.22)
block(11.67,8.25,1.12,2.0,purple,'43',8.3,depth=.22)
block(13.79,8.5,.65,1.5,teal,'2',8.3,depth=.18)
arrow([(8.31,9.2),(9.51,9.2)])
arrow([(10.88,9.2),(11.66,9.2)])
arrow([(13.03,9.2),(13.78,9.2)])
txt(10.14,7.72,'Hidden 1',6.8)
txt(12.28,7.72,'Hidden 2',6.8)
txt(11.3,6.67,'Linear → LayerNorm\nReLU → Dropout (0.10)',7.5)
txt(14.1,7.89,'Linear',6.3)
txt(14.1,7.41,r'$43\rightarrow2$',6.6)
txt(14.38,11.42,'315 → 43 → 43 → 2   |   15,740 parameters',7.5)

box(15.24,9.76,1.75,.74,'raw_mu',face='#DFECF8',size=7.5)
box(15.24,7.98,1.75,.74,'logit_high',face='#DFECF8',size=7.5)
arrow([(14.62,9.4),(14.94,9.4),(14.94,10.13),(15.22,10.13)])
arrow([(14.62,9.0),(14.94,9.0),(14.94,8.35),(15.22,8.35)])
arrow([(17.04,10.13),(18.06,10.13)])
txt(17.62,10.63,'Softplus',6.2)
txt(18.54,10.14,r'$\mu^{+}$',9,'bold')
txt(18.57,9.39,'Continuous\noutput',6.25)
arrow([(17.04,8.35),(18.06,8.35)])
txt(18.59,8.38,'High-regret\nlogit',6.25)

box(.73,1.85,2.64,1.25,'True regret\n'+r'$r_g$',face='#F1C2AD',size=7.5,edge=orange)
box(4.31,2.88,3.8,.91,r'$y_g^{reg}=\log(1+r_g/h)$',face='white',size=7.5,edge=orange)
box(4.31,1.2,3.8,.91,r'$y_g^{high}=\mathbf{1}[r_g>h]$',face='white',size=7.5,edge=orange)
arrow([(3.43,2.7),(3.8,2.7),(3.8,3.33),(4.25,3.33)],color=orange)
arrow([(3.43,2.2),(3.8,2.2),(3.8,1.66),(4.25,1.66)],color=orange)
txt(6.2,.76,r'Frozen threshold $h=0.01$',6.5)
box(9.13,1.3,6.15,2.48,face='#FBE7DD',edge=orange)
txt(12.2,3.25,'Joint training loss',7.5,'bold')
txt(12.2,2.67,r'$\mathcal{L}=\mathrm{SmoothL1}(\mu^{+},y_g^{reg})$',7.5)
txt(12.2,2.08,r'$+\ 0.25\,\mathrm{BCEWithLogits}_{w}(\mathrm{logit\_high},y_g^{high})$',6.7)
txt(12.2,1.58,'Training-fitted class weight',7.5)
arrow([(8.18,3.33),(9.08,3.33)],dashed=True,color=orange)
arrow([(8.18,1.66),(9.08,1.66)],dashed=True,color=orange)
arrow([(18.69,9.87),(19.6,9.87),(19.6,3.34),(15.33,3.34)],dashed=True,color=orange)
arrow([(18.69,8.0),(18.0,8.0),(18.0,2.0),(15.33,2.0)],dashed=True,color=orange)
txt(17.6,1.02,'Training only',7.5,color=orange)

fig.savefig(ROOT/'Direct_audit_head_architecture.svg',facecolor='white')
fig.savefig(ROOT/'Direct_audit_head_architecture.pdf',facecolor='white')
fig.savefig(ROOT/'Direct_audit_head_architecture.png',dpi=600,facecolor='white')
plt.close(fig)
print('Saved architecture figure: SVG, PDF, PNG')
