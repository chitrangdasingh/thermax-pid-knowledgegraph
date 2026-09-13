from pathlib import Path
import json
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Thermax P&ID Intelligence",page_icon="⚙️",layout="wide")
ROOT=Path(__file__).parent
DATA=ROOT/"data"

@st.cache_data
def load_data():
    assets=pd.read_csv(DATA/"asset_master.csv").fillna("")
    rels=pd.read_csv(DATA/"graph_relationships.csv").fillna("")
    hierarchy=pd.read_csv(DATA/"asset_hierarchy.csv").fillna("")
    review=pd.read_csv(DATA/"validation_queue.csv").fillna("")
    summary=json.loads((DATA/"platform_summary.json").read_text())
    return assets,rels,hierarchy,review,summary

assets,rels,hierarchy,review,summary=load_data()

def asset_label(row):
    return str(row.tag) if str(row.tag).strip() else str(row.asset_id)

def connection_graph():
    g=nx.Graph()
    for _,r in assets.iterrows(): g.add_node(str(r.asset_id),tag=str(r.tag),asset_class=str(r.asset_class))
    for _,r in rels[rels.relationship=="CONNECTED_TO"].iterrows(): g.add_edge(str(r.source_id),str(r.target_id),confidence=float(r.confidence))
    return g

def node_for_label(label):
    found=assets[(assets.tag.astype(str)==label)|(assets.asset_id.astype(str)==label)]
    return str(found.iloc[0].asset_id) if len(found) else None

st.title("Thermax P&ID Knowledge Graph PoC")
st.caption("DXF ingestion · confidence-scored BOM · asset hierarchy · graph queries · engineer validation")
cols=st.columns(6)
for c,(value,label) in zip(cols,[(summary["asset_candidates"],"Asset candidates"),(summary["tagged_candidates"],"Unique tags"),(summary["native_blocks"],"Native blocks"),(summary["graph_nodes"],"Graph nodes"),(summary["graph_edges"],"Graph edges"),(summary["pending_reviews"],"Pending review")]): c.metric(label,value)
st.warning("This is a PoC. Rule-based associations and inferred connections are not approved engineering truth until reviewed.")

page=st.sidebar.radio("Workspace",["Ingestion Pipeline","P&ID Explorer","Graph Q&A","BOM & Hierarchy","Validation","Data Readiness"])
areas=sorted(assets.area.unique())
selected_areas=st.sidebar.multiselect("Areas",areas,default=areas)
levels=st.sidebar.multiselect("Confidence",["HIGH","MEDIUM","LOW"],default=["HIGH","MEDIUM","LOW"])
view=assets[assets.area.isin(selected_areas)&assets.confidence_level.isin(levels)].copy()

if page=="Ingestion Pipeline":
    st.subheader("From engineering drawing to structured plant intelligence")
    stages=pd.DataFrame([
        {"Stage":"1 · Ingest","Method":"DXF vector reader + drawing registry","Generated evidence":"13,985 CAD entities · 10 sheets","Status":"COMPLETE"},
        {"Stage":"2 · Text/OCR","Method":"Native DXF TEXT/MTEXT first; OCR fallback for scans","Generated evidence":"847 text records · 24 unique asset/interlock tags after filtering","Status":"COMPLETE_FOR_DXF"},
        {"Stage":"3 · Vision AI","Method":"Native symbol library + plant-specific YOLO/FPN detector","Generated evidence":"49 native CAD blocks; CNN package ready","Status":"CNN_AWAITS_REVIEWED_LABELS"},
        {"Stage":"4 · Classify","Method":"Tag convention + spatial association + confidence provenance","Generated evidence":"73 asset candidates with element-level scores","Status":"COMPLETE_REVIEW_REQUIRED"},
        {"Stage":"5 · Topology","Method":"Line-component reconstruction + graph rules","Generated evidence":"26 preliminary connections","Status":"ENGINEER_REVIEW_REQUIRED"},
        {"Stage":"6 · Structure","Method":"Asset hierarchy + unified namespace + knowledge graph","Generated evidence":"97 graph nodes · 191 relationships","Status":"COMPLETE_REVIEW_REQUIRED"},
    ])
    st.dataframe(stages,width="stretch",hide_index=True)
    st.progress(1.0,text="Ingestion complete")
    st.progress(1.0,text="DXF text extraction complete")
    st.progress(49/73,text="Native symbol evidence coverage (49 of 73 candidates)")
    st.progress(0.0,text="Plant-specific CNN training awaits reviewed labels")
    st.progress(0.0,text="Final topology approval awaits engineer verification")
    st.caption("OCR reads text; P&ID digitization additionally identifies symbols, binds tags, reconstructs topology, verifies each element, and maps approved entities into the plant hierarchy.")

elif page=="P&ID Explorer":
    left,right=st.columns([1.7,1])
    with left:
        st.subheader("Zoomable P&ID with detected assets")
        fig=px.scatter(view,x="x",y="y",color="confidence_level",symbol="symbol_source",hover_name="tag",
            hover_data=["asset_id","asset_class","area","overall_confidence","confidence_basis","review_status"],
            color_discrete_map={"HIGH":"#16a34a","MEDIUM":"#f59e0b","LOW":"#dc2626"})
        image_path=DATA/"pid_render.png"
        if image_path.exists():
            bg=Image.open(image_path)
            fig.add_layout_image(dict(source=bg,xref="x",yref="y",x=80,y=303,sizex=4980,sizey=303,sizing="stretch",opacity=.58,layer="below",xanchor="left",yanchor="top"))
            fig.update_xaxes(range=[80,5060]); fig.update_yaxes(range=[0,303])
        fig.update_traces(marker={"size":11,"line":{"width":1,"color":"white"}})
        fig.update_layout(height=640,template="plotly_white",dragmode="zoom",yaxis_scaleanchor="x",legend_orientation="h")
        event=st.plotly_chart(fig,width="stretch",on_select="rerun",selection_mode="points",key="pid")
    with right:
        st.subheader("Selected BOM result")
        selected=None
        if event and event.selection.points:
            point=event.selection.points[0]
            idx=point.get("point_index",point.get("point_number"))
            if idx is not None and idx < len(view): selected=view.iloc[int(idx)]
        if selected is None:
            label=st.selectbox("Or select a tag/asset",[asset_label(r) for _,r in view.iterrows()]) if len(view) else None
            if label: selected=view[[asset_label(r)==label for _,r in view.iterrows()]].iloc[0]
        if selected is not None:
            st.metric("Overall confidence",f"{float(selected.overall_confidence):.1%}",selected.confidence_level)
            fields={"Tag":selected.tag or "Unassigned","Asset ID":selected.asset_id,"Class":selected.asset_class,"Area":selected.area,"Evidence":selected.symbol_source,"Confidence basis":selected.confidence_basis,"Review status":selected.review_status}
            st.dataframe(pd.DataFrame(fields.items(),columns=["Field","Result"]),hide_index=True,width="stretch")

elif page=="Graph Q&A":
    st.subheader("Structured questions against the P&ID knowledge graph")
    qtype=st.selectbox("Question type",["Simple counting","Direct connection (single-hop)","Connection path (multi-hop)","Tag prefix / value"])
    g=connection_graph(); labels=sorted(set(asset_label(r) for _,r in assets.iterrows()))
    if qtype=="Simple counting":
        cls=st.selectbox("Asset class",sorted(assets.asset_class.unique()))
        result=assets[assets.asset_class==cls]
        st.success(f"{len(result)} symbols/components of class '{cls}' are present.")
        st.dataframe(result[["asset_id","tag","area","overall_confidence","review_status"]],hide_index=True,width="stretch")
    elif qtype=="Direct connection (single-hop)":
        label=st.selectbox("Selected tag or asset",labels); node=node_for_label(label)
        neighbours=list(g.neighbors(node)) if node in g else []
        result=assets[assets.asset_id.astype(str).isin(neighbours)]
        st.success(f"{len(result)} directly connected candidates found for {label}.")
        st.dataframe(result[["asset_id","tag","asset_class","overall_confidence","review_status"]],hide_index=True,width="stretch")
    elif qtype=="Connection path (multi-hop)":
        a=st.selectbox("From",labels,index=0); b=st.selectbox("To",labels,index=min(1,len(labels)-1)); na=node_for_label(a); nb=node_for_label(b)
        try:
            path=nx.shortest_path(g,na,nb); result=assets[assets.asset_id.astype(str).isin(path)].copy(); order={n:i for i,n in enumerate(path)}; result["path_order"]=result.asset_id.astype(str).map(order); result=result.sort_values("path_order")
            st.success(" → ".join(result.apply(asset_label,axis=1))); st.dataframe(result[["path_order","asset_id","tag","asset_class","review_status"]],hide_index=True,width="stretch")
        except Exception: st.info("No validated/inferred connection path is available between these selections yet.")
    else:
        prefix=st.text_input("Tag prefix",value="HE").strip().upper()
        result=assets[assets.tag.astype(str).str.upper().str.startswith(prefix)]
        st.success(f"{len(result)} matching tags found.")
        st.dataframe(result[["tag","asset_class","area","overall_confidence","review_status"]],hide_index=True,width="stretch")

elif page=="BOM & Hierarchy":
    tab1,tab2,tab3=st.tabs(["Asset Master","Hierarchy","Relationships"])
    with tab1:
        st.plotly_chart(px.bar(view.asset_class.value_counts().rename_axis("class").reset_index(name="count"),x="class",y="count",color="class"),width="stretch")
        st.dataframe(view,width="stretch",height=480); st.download_button("Download BOM",assets.to_csv(index=False).encode(),"asset_master.csv")
    with tab2: st.dataframe(hierarchy,width="stretch",hide_index=True)
    with tab3: st.dataframe(rels,width="stretch",height=550); st.download_button("Download relationships",rels.to_csv(index=False).encode(),"graph_relationships.csv")

elif page=="Validation":
    st.subheader("Engineer review queue")
    edited=st.data_editor(review,width="stretch",height=600,num_rows="dynamic",disabled=["asset_id","overall_confidence","confidence_level"])
    st.download_button("Download reviewed validation log",edited.to_csv(index=False).encode(),"validation_queue_reviewed.csv")

else:
    st.subheader("Integration readiness")
    readiness=pd.DataFrame({"Layer":["DXF ingestion","Asset master","Knowledge graph","Engineer validation","CNN symbol model","Historian time-series","Maintenance history","KPI definitions"],"Status":["Generated","Generated","Generated - topology review needed","Queue generated","Awaiting reviewed labels","Schema only","Schema only","Schema only"]})
    st.dataframe(readiness,width="stretch",hide_index=True)
    st.info("Predictive maintenance and process-performance claims remain disabled until real time-series, CMMS and KPI data are connected.")
