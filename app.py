"""Run with: streamlit run app.py"""
from pathlib import Path
import base64
import json
import os
import secrets

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import networkx as nx

from platform_core import DATA, DB, add_derived_channel, add_manual_asset, asset_context, approve_mapping, bootstrap, compute_derived, evaluate_rules, ingest_observations, namespaces, read_table, record_review, set_hierarchy, summarize_asset, upsert_mapping
from analytics import advise, heat_duty, train_forecaster, trend, series

st.set_page_config(page_title="Thermax P&ID Digital Twin POC",layout="wide",page_icon="⚙️")


@st.cache_resource
def initialise():
    return bootstrap()


try:
    summary=initialise()
except Exception as exc:
    st.error(f"Could not initialise the bundled DXF data: {exc}")
    st.stop()

assets=read_table("assets"); rels=read_table("relationships"); blocks=read_table("cad_blocks")
bounds=json.loads((DATA/"drawing_bounds.json").read_text())
st.title("Thermax P&ID · Digital Twin POC")
st.caption("Single Tube Reactor System | source: supplied DXF | evidence and provenance shown per record")
st.info("The P&ID gives drawing structure. Live values, predictions and alerts activate only after an engineer approves a historian tag mapping and measured data are imported. Class and connection confidence scores are heuristics, not trained CNN probabilities.")
pages=["Drawing explorer","Vision AI inspection","Asset registry / BOM","Hierarchy & templates","Knowledge graph","Graph-RAG Q&A","Azure ingestion","Data connectors","Analytics & physics","Performance cortex","Model registry","Alerts & workflows","Copilot & API","Audit & exports"]
page=st.sidebar.radio("Workspace",pages)
st.sidebar.caption(f"DXF entities: {summary['entity_count']:,} | assets: {len(assets)} | candidate edges: {len(rels)}")
review_token=os.getenv("PID_REVIEW_TOKEN","")
input_token=st.sidebar.text_input("Engineering review token",type="password") if review_token else ""
review_unlocked=bool(review_token and input_token and secrets.compare_digest(review_token,input_token))


def asset_panel(aid):
    ctx=asset_context(aid)
    if ctx is None:return
    a=ctx["asset"]
    st.subheader(f"{a['tag']} · {a['asset_class']}")
    c=st.columns(4)
    for item,(title,value) in zip(c,[("DXF evidence score",f"{a['overall_confidence']:.1%}"),
        ("Evidence coverage",f"{a['evidence_coverage']:.1%}"),("Twin readiness",f"{a['digital_twin_readiness']:.1%}"),
        ("Latest observations",len(ctx['observations']))]):item.metric(title,value)
    st.caption(f"Source: {a['provenance']} · CAD layer: {a['cad_layer']} · Review: {a['review_status']}")
    left,right=st.columns(2)
    with left:
        st.markdown("**Candidate relationships**")
        st.dataframe(pd.DataFrame(ctx["relationships"]),hide_index=True,use_container_width=True)
        st.markdown("**Mapped sensors**")
        st.dataframe(pd.DataFrame(ctx["mappings"]),hide_index=True,use_container_width=True)
    with right:
        st.markdown("**Recent measured values**")
        st.dataframe(pd.DataFrame(ctx["observations"]),hide_index=True,use_container_width=True)
        st.markdown("**Alerts**")
        st.dataframe(pd.DataFrame(ctx["alerts"]),hide_index=True,use_container_width=True)
    st.caption("A purchasing BOM requires approved equipment specifications, material, size, make, model, quantity and revision; this is a DXF asset register.")


if page=="Drawing explorer":
    st.caption("Zoom/pan the actual flattened vector rendering. Click a colored asset marker; CAD lines and blocks can also be looked up by ID below.")
    query=st.text_input("Search tags or class",key="drawing_query")
    inspect_lines=st.checkbox("Enable clicking on vector segment midpoints (53,958 drawing primitives)",value=False)
    shown=assets[assets[["tag","asset_class","asset_id"]].astype(str).apply(lambda col:col.str.contains(query,case=False,regex=False)).any(axis=1)] if query else assets
    fig=go.Figure()
    png=base64.b64encode((DATA/"vector_drawing.png").read_bytes()).decode()
    fig.add_layout_image(dict(source="data:image/png;base64,"+png,xref="x",yref="y",x=bounds["xmin"],y=bounds["ymax"],
        sizex=bounds["xmax"]-bounds["xmin"],sizey=bounds["ymax"]-bounds["ymin"],sizing="stretch",layer="below"))
    lookup=assets.set_index("asset_id")
    for r in rels.itertuples():
        if r.source_id in lookup.index and r.target_id in lookup.index:
            ends=lookup.loc[[r.source_id,r.target_id]]
            fig.add_trace(go.Scatter(x=ends.x,y=ends.y,mode="lines",line=dict(color="rgba(234,162,25,.55)",width=1,dash="dot"),hoverinfo="skip",showlegend=False))
    line_trace=None
    if inspect_lines:
        segment_points=read_table("line_segments",limit=60000)
        line_trace=len(fig.data)
        fig.add_trace(go.Scattergl(x=(segment_points.x1+segment_points.x2)/2,
            y=(segment_points.y1+segment_points.y2)/2,mode="markers",customdata=segment_points.segment_id,
            marker=dict(size=5,color="rgba(232,107,24,.42)"),name="Vector midpoints",
            hovertemplate="Segment %{customdata}<extra></extra>"))
    points=shown.reset_index(drop=True)
    fig.add_trace(go.Scatter(x=points.x,y=points.y,mode="markers+text",text=points.tag,textposition="top center",
        customdata=points[["asset_id","asset_class","overall_confidence"]].to_numpy(),
        hovertemplate="%{text}<br>%{customdata[1]}<br>Evidence: %{customdata[2]:.1%}<extra></extra>",
        marker=dict(size=13,color="#067fa4",line=dict(color="white",width=1.5)),name="Assets"))
    fig.update_layout(height=620,margin=dict(l=5,r=5,t=5,b=5),plot_bgcolor="white",dragmode="pan",clickmode="event+select",
        xaxis=dict(range=[bounds["xmin"],bounds["xmax"]],showgrid=False),
        yaxis=dict(range=[bounds["ymin"],bounds["ymax"]],showgrid=False,scaleanchor="x"))
    event=st.plotly_chart(fig,use_container_width=True,on_select="rerun",selection_mode="points",key="drawing")
    selected=None
    selected_segment=None
    if event and event.selection.points:
        for p in event.selection.points:
            if p.get("curve_number")==len(fig.data)-1 and p.get("point_index") is not None:
                selected=points.iloc[p["point_index"]].asset_id;break
            if line_trace is not None and p.get("curve_number")==line_trace and p.get("point_index") is not None:
                selected_segment=segment_points.iloc[p["point_index"]].segment_id
    if selected_segment:
        st.success(f"Clicked vector segment {selected_segment} — drawing geometry, pipe classification pending")
        st.dataframe(read_table("line_segments",where="WHERE segment_id=?",params=(selected_segment,),limit=1),hide_index=True,use_container_width=True)
    if not points.empty:
        index=points.asset_id.tolist().index(selected) if selected in points.asset_id.tolist() else 0
        selected=st.selectbox("Selected asset (also works if chart click is unavailable)",points.asset_id.tolist(),index=index,
            format_func=lambda a:f"{a} · {lookup.loc[a,'tag']}")
        asset_panel(selected)
    st.divider()
    l,r=st.columns(2)
    with l:
        st.subheader("Drawing line primitive")
        st.caption("Each polyline segment is drawable geometry. It is not yet a validated pipe or tagged line.")
        segment_id=st.text_input("Segment ID",value="SEG-000001")
        row=read_table("line_segments",where="WHERE segment_id=?",params=(segment_id.strip().upper(),),limit=1)
        if len(row):st.dataframe(row,hide_index=True,use_container_width=True)
        else:st.write("No matching segment")
    with r:
        st.subheader("CAD block insert")
        if len(blocks):
            block=st.selectbox("Block ID",blocks.block_id.tolist())
            st.dataframe(blocks[blocks.block_id.eq(block)],hide_index=True,use_container_width=True)
        else:st.caption("No INSERT entities found")

elif page=="Vision AI inspection":
    st.subheader("YOLOv5 symbol detection + Tesseract OCR + association")
    st.caption("Box convention: green tagged asset, red symbol without tag, blue tag without symbol, orange item with a review flag. Text and symbols are associated by bounded one-to-one proximity. Custom engineer-approved P&ID weights are mandatory.")
    slide=st.file_uploader("Upload P&ID page image",type=["png","jpg","jpeg"],key="vision_slide")
    weights=st.file_uploader("Upload approved custom YOLO weights (best.pt)",type=["pt"],key="vision_weights")
    threshold=st.slider("Minimum symbol confidence",.05,.90,.12,.01)
    if slide and weights and st.button("Run vision inspection"):
        try:
            import tempfile
            from pid_vision import run as run_vision
            with tempfile.TemporaryDirectory() as temp:
                folder=Path(temp);image_path=folder/slide.name;weight_path=folder/"best.pt";result_dir=folder/"results"
                image_path.write_bytes(slide.getvalue());weight_path.write_bytes(weights.getvalue())
                vision_summary=run_vision(image_path,weight_path,result_dir,conf=threshold);st.json(vision_summary)
                st.image(str(result_dir/"annotated_pid.png"),caption="Detected symbols, OCR associations and review flags",use_container_width=True)
                detection=pd.read_csv(result_dir/"detection_results.csv");flaws=pd.read_csv(result_dir/"flaw_report.csv");vision_hierarchy=pd.read_csv(result_dir/"asset_hierarchy.csv")
                st.markdown("**Detection and confidence table**")
                st.dataframe(detection,hide_index=True,use_container_width=True)
                st.markdown("**Visible flaw/review table**")
                st.dataframe(flaws,hide_index=True,use_container_width=True)
                st.markdown("**Generated asset hierarchy**")
                st.dataframe(vision_hierarchy,hide_index=True,use_container_width=True)
                st.download_button("Download detection table",detection.to_csv(index=False),"detection_results.csv")
                st.download_button("Download flaw table",flaws.to_csv(index=False),"flaw_report.csv")
                st.download_button("Download hierarchy",vision_hierarchy.to_csv(index=False),"asset_hierarchy.csv")
        except ImportError:st.error("Vision dependencies are not installed. Use requirements-vision.txt.")
        except Exception as exc:st.error(f"Inspection failed: {exc}")
    st.info("No approved best.pt is bundled. Use cnn_preannotations/ in CVAT, correct every box/class across several P&IDs, add hard negatives, export a drawing-level train/val/test split, and then run train_yolov5.py. The 24 current boxes are weak preannotations, not a trained-CNN result.")

elif page=="Asset registry / BOM":
    st.subheader("DXF asset register")
    st.dataframe(assets,hide_index=True,use_container_width=True,height=540)
    st.download_button("Download asset CSV",assets.to_csv(index=False).encode(),"asset_master.csv","text/csv")
    if len(blocks):
        st.subheader("CAD blocks · unclassified")
        st.caption("All 49 INSERT entities in this DXF fall outside the rendered process drawing and are likely legend/library symbols; they are not counted as installed assets.")
        st.dataframe(blocks,hide_index=True,use_container_width=True)
    st.warning("Drawing text and geometry do not provide verified purchase specifications or asset quantity. Request vendor BOM and equipment datasheets.")

elif page=="Hierarchy & templates":
    st.subheader("Enterprise → Site → Area → Process unit → Asset")
    st.caption("Initial organizational labels are draft placeholders. Engineering can place assets into their approved plant hierarchy.")
    hierarchy=read_table("hierarchy")
    with st.expander("P&ID drawing library"):
        st.dataframe(read_table("drawings"),hide_index=True,use_container_width=True)
        st.caption("One drawing/revision is bundled. Cross-drawing reconciliation requires the approved drawing index, revisions and shared tag dictionary.")
    st.dataframe(hierarchy,hide_index=True,use_container_width=True,height=420)
    unit=st.selectbox("Explore process unit",sorted(hierarchy.process_unit.unique().tolist()))
    st.dataframe(hierarchy[hierarchy.process_unit.eq(unit)].merge(assets[["asset_id","tag","asset_class"]],on="asset_id"),hide_index=True,use_container_width=True)
    if review_unlocked:
        with st.expander("Add custom asset with a drawing location"):
            with st.form("manual"):
                tag=st.text_input("New asset tag");kind=st.text_input("Engineering class")
                x=st.number_input("DXF x coordinate",value=1000.0);y=st.number_input("DXF y coordinate",value=100.0)
                reviewer=st.text_input("Reviewer",key="manual_reviewer")
                if st.form_submit_button("Create draft asset"):
                    try:st.success("Created "+add_manual_asset(tag,kind,x,y,reviewer)+". Refresh to see it on the drawing.")
                    except ValueError as exc:st.error(str(exc))
        with st.expander("Place asset into reviewed hierarchy"):
            with st.form("hierarchy"):
                aid=st.selectbox("Asset",assets.asset_id.tolist())
                enterprise=st.text_input("Enterprise",value="Thermax")
                site=st.text_input("Site",value="RTIC")
                area=st.text_input("Area",value="Single Tube Reactor")
                process_unit=st.text_input("Process unit")
                reviewer=st.text_input("Reviewer",key="hier_reviewer")
                if st.form_submit_button("Save reviewed hierarchy"):
                    try:set_hierarchy(aid,enterprise,site,area,process_unit,reviewer);st.success("Hierarchy saved")
                    except ValueError as exc:st.error(str(exc))
    st.info("Reusable sensor templates require approved I/O lists and units. The virtual difference channel below is a limited example; it never infers sensors from geometry.")
    mappings=read_table("tag_mappings");approved=mappings[mappings.engineer_status.eq("APPROVED")].sensor_tag.tolist() if len(mappings) else []
    if len(approved)>=2:
        with st.form("derived"):
            name=st.text_input("Virtual channel ID",value="TEMP_DELTA_01")
            first=st.selectbox("Input A",approved);second=st.selectbox("Input B",approved,index=1)
            if st.form_submit_button("Define A minus B"):
                try:add_derived_channel(name,first,second);st.success("Draft virtual channel created")
                except Exception as exc:st.error(str(exc))
    derived=read_table("derived_channels")
    if len(derived):
        st.dataframe(derived,hide_index=True,use_container_width=True)
        channel=st.selectbox("Evaluate virtual channel",derived.channel_id.tolist())
        st.json(compute_derived(channel))

elif page=="Knowledge graph":
    g=nx.Graph();g.add_node("DRAWING-001",label="DXF",kind="drawing")
    classes=assets.asset_class.unique().tolist()
    for c in classes:g.add_node("CLASS:"+c,label=c,kind="class")
    for a in assets.itertuples():
        g.add_node(a.asset_id,label=a.tag,kind="asset")
        g.add_edge("DRAWING-001",a.asset_id,kind="SHOWN_ON")
        g.add_edge("CLASS:"+a.asset_class,a.asset_id,kind="TYPE_OF")
    for edge in rels.itertuples():g.add_edge(edge.source_id,edge.target_id,kind="CANDIDATE")
    pos=nx.spring_layout(g,seed=21,iterations=100)
    fig=go.Figure()
    for kind,color in [("SHOWN_ON","#64748b"),("TYPE_OF","#1c6491"),("CANDIDATE","#e9a21e")]:
        x=[];y=[]
        for u,v,d in g.edges(data=True):
            if d["kind"]==kind:
                x.extend([pos[u][0],pos[v][0],None]);y.extend([pos[u][1],pos[v][1],None])
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(color=color,width=1.4,dash="dot" if kind=="CANDIDATE" else "solid"),name=kind,hoverinfo="skip"))
    for kind,color,size in [("drawing","#2563eb",27),("class","#06b6d4",17),("asset","#20c997",10)]:
        ids=[n for n,d in g.nodes(data=True) if d["kind"]==kind]
        fig.add_trace(go.Scatter(x=[pos[n][0] for n in ids],y=[pos[n][1] for n in ids],text=[g.nodes[n]["label"] for n in ids],mode="markers+text",
            textposition="top center",marker=dict(size=size,color=color),name=kind,hovertext=ids))
    fig.update_layout(height=620,template="plotly_dark",xaxis=dict(visible=False),yaxis=dict(visible=False),margin=dict(l=5,r=5,t=5,b=5))
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f"Exported graph: {summary['graph_nodes']} nodes, {summary['graph_edges']} edges. This simplified display uses undirected candidate links; review direction in the data table.")
    selected=st.selectbox("Explore asset",assets.asset_id.tolist(),format_func=lambda a:f"{a} · {assets.set_index('asset_id').loc[a,'tag']}")
    asset_panel(selected)
    st.download_button("Download GraphML",(DATA/"knowledge_graph.graphml").read_bytes(),"knowledge_graph.graphml")

elif page=="Graph-RAG Q&A":
    from dexpi_graphrag import condensed_graph, gemini_answer, graph_context, load_current_graph, load_dexpi_proteus, local_answer
    st.subheader("Searchable P&ID knowledge graph and Q&A")
    st.caption("The current high-level graph is derived from the flattened DXF and remains a proposed DEXPI mapping. Domain connections and lexical similarities stay separate. Answers cite graph evidence and preserve candidate/review status.")
    graph=condensed_graph(load_current_graph(DATA),include_lexical=True)
    domain_edges=sum(1 for *_,edge in graph.edges(data=True) if edge.get("relation_group")!="lexical")
    lexical_edges=graph.number_of_edges()-domain_edges
    c1,c2,c3=st.columns(3);c1.metric("Asset candidates",graph.number_of_nodes());c2.metric("Candidate domain links",domain_edges);c3.metric("Lexical links",lexical_edges)
    examples=["How many reactors are present?","What is connected to R-300?","What data are needed to calculate methanol conversion?","Give safety review prompts for this reactor system"]
    example=st.selectbox("Example question",examples)
    with st.form("graph_question"):
        question=st.text_input("Ask the P&ID",value=example)
        ask=st.form_submit_button("Ask local evidence engine")
    if ask:
        answer_result=local_answer(question,graph);st.session_state["graph_answer"]=answer_result;st.session_state["graph_question_text"]=question
    if "graph_answer" in st.session_state:
        result=st.session_state["graph_answer"]
        st.success(result["answer"])
        st.caption(f"Answer type: {result.get('answer_type')} · status: {result.get('review_status')}")
        if result.get("connections"):st.dataframe(pd.DataFrame(result["connections"]),hide_index=True,use_container_width=True)
        if result.get("evidence"):st.dataframe(pd.DataFrame(result["evidence"]),hide_index=True,use_container_width=True)
        context=graph_context(graph,st.session_state.get("graph_question_text",question))
        with st.expander("Bounded graph context used for optional Graph-RAG"):
            st.json(context)
        if st.button("Ask configured Gemini model with this evidence"):
            try:st.json(gemini_answer(st.session_state.get("graph_question_text",question),context))
            except Exception as exc:st.error(f"Gemini adapter is inactive: {exc}")
    st.divider()
    st.markdown("**Optional true DEXPI input**")
    st.caption("Upload a licensed DEXPI 1.3 Proteus XML to build plant, process and conceptual graphs with pyDEXPI. A DXF renamed to XML is not valid.")
    dexpi_upload=st.file_uploader("DEXPI Proteus XML",type=["xml"],key="dexpi_xml")
    if dexpi_upload and st.button("Load DEXPI graph"):
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as temp:
                target=Path(temp)/"uploaded.xml";target.write_bytes(dexpi_upload.getvalue())
                graphs=load_dexpi_proteus(target)
                st.dataframe(pd.DataFrame([{"graph":name,"nodes":g.number_of_nodes(),"edges":g.number_of_edges(),"dexpi_status":g.graph.get("dexpi_status")} for name,g in graphs.items()]),hide_index=True,use_container_width=True)
        except Exception as exc:st.error(f"DEXPI load failed: {exc}")

elif page=="Azure ingestion":
    st.subheader("Microsoft reference P&ID digitization workflow")
    st.caption("Alternative cloud path using the Azure Samples training and inference repositories. The service must first be deployed in a Thermax-approved Azure environment; this page only calls an existing HTTPS endpoint.")
    st.code("Symbol detection → manual correction → text detection → manual correction → graph construction → manual correction → graph persistence",language="text")
    base_url=st.text_input("Azure inference service URL",value=os.getenv("AZURE_PID_SERVICE_URL",""),placeholder="https://private-service.example")
    api_key=st.text_input("API key (kept only in this app session)",type="password")
    pid_id=st.text_input("Unique P&ID run ID",value="thermax-str-review-001")
    azure_image=st.file_uploader("High-resolution P&ID PNG/JPEG",type=["png","jpg","jpeg"],key="azure_pid_image")
    box_text=st.text_input("Normalized diagram bounding box",value='{"topX":0.0,"topY":0.0,"bottomX":1.0,"bottomY":1.0}')
    if azure_image and st.button("Run Azure symbol stage"):
        try:
            import tempfile
            from azure_pid_client import AzurePidClient
            with tempfile.TemporaryDirectory() as temp:
                image_path=Path(temp)/azure_image.name;image_path.write_bytes(azure_image.getvalue())
                response=AzurePidClient(base_url=base_url,api_key=api_key or None).symbol_detection(pid_id,image_path,json.loads(box_text))
                st.session_state["azure_symbol_response"]=response
        except Exception as exc:st.error(f"Azure symbol stage failed: {exc}")
    if "azure_symbol_response" in st.session_state:
        st.json(st.session_state["azure_symbol_response"])
        st.download_button("Download and manually review symbol JSON",json.dumps(st.session_state["azure_symbol_response"],indent=2),"azure_symbol_response.json","application/json")
    st.warning("Do not submit the next stage until an engineer has corrected/approved the preceding JSON. Graph construction may be asynchronous in the Microsoft reference service.")
    with st.expander("Submit a reviewed JSON to a later stage"):
        stage=st.selectbox("Stage",["text","graph","persist"])
        reviewed_payload=st.text_area("Reviewed JSON",height=240)
        if st.button("Submit reviewed JSON"):
            try:
                from azure_pid_client import AzurePidClient
                st.json(AzurePidClient(base_url=base_url,api_key=api_key or None).post_stage(stage,pid_id,json.loads(reviewed_payload)))
            except Exception as exc:st.error(f"Azure {stage} stage failed: {exc}")
    st.info("See docs/AZURE_REFERENCE_IMPLEMENTATION.md and azure_reference_config.yaml for the deployment and normalization plan.")

elif page=="Data connectors":
    st.subheader("Unified namespace and historian connector")
    st.caption("Only an engineer-approved tag-to-asset mapping may ingest observations. Cloud file uploads exist only in that app session unless you configure durable storage.")
    l,r=st.columns(2)
    with l:
        with st.form("mapping"):
            tag=st.text_input("Historian sensor tag",placeholder="TT_300_PV")
            aid=st.selectbox("P&ID asset ID",assets.asset_id.tolist(),format_func=lambda a:f"{a} · {assets.set_index('asset_id').loc[a,'tag']}")
            prop=st.selectbox("Measured property",["temperature","pressure","flow","level","composition","power","other"])
            unit=st.text_input("Engineering unit",value="degC")
            low=st.number_input("Low alert threshold",value=0.0)
            high=st.number_input("High alert threshold",value=100.0)
            submit=st.form_submit_button("Create pending mapping")
        if submit:
            try:
                if low>=high:raise ValueError("Low threshold must be below high")
                upsert_mapping(tag.strip(),aid,prop,unit.strip(),low,high)
                st.success("Mapping stored as PENDING. Engineering approval required before ingestion.")
            except ValueError as exc:st.error(str(exc))
    with r:
        mappings=read_table("tag_mappings")
        st.dataframe(mappings,hide_index=True,use_container_width=True)
        st.caption("Namespace pattern: thermax/rtic/reactor_pid/{asset_id}/{property}")
        st.json(namespaces())
    if review_unlocked and len(mappings):
        with st.form("approve_mapping"):
            pending=mappings[mappings.engineer_status.ne("APPROVED")]
            if len(pending):
                pending_tag=st.selectbox("Approve tag mapping",pending.sensor_tag.tolist())
                reviewer=st.text_input("Engineering reviewer name")
                if st.form_submit_button("Approve mapping"):
                    try:approve_mapping(pending_tag,reviewer);st.success("Mapping approved. Refresh to view status.")
                    except ValueError as exc:st.error(str(exc))
    elif not review_token:st.caption("Approval controls are locked: set PID_REVIEW_TOKEN in the deployment's server environment.")
    st.subheader("Import measured historian CSV")
    st.code("timestamp,sensor_tag,asset_id,value,unit,quality_code\n2026-09-22T09:00:00+05:30,TT_300_PV,AST-0003,220.0,degC,GOOD",language="text")
    file=st.file_uploader("Choose CSV with REAL readings",type="csv")
    if file and st.button("Validate and import readings"):
        try:st.json(ingest_observations(pd.read_csv(file)))
        except Exception as exc:st.error(str(exc))
    st.caption("REST batch connector: POST /observations with PID_API_KEY. Data owners can adapt OPC UA/MQTT to the same validated schema behind a company gateway.")

elif page=="Analytics & physics":
    st.subheader("Measured-data analysis")
    mapping=read_table("tag_mappings")
    if mapping.empty:st.warning("No historian tag mapping yet. Predictive analytics is inactive.")
    else:
        chosen=st.selectbox("Mapped sensor",mapping.sensor_tag.tolist())
        df=series(chosen);st.json(trend(chosen))
        if len(df):st.line_chart(df.set_index("timestamp").value)
        if st.button("Train one-step forecast on measured data"):
            try:st.json(train_forecaster(chosen))
            except Exception as exc:st.error(str(exc))
        st.caption("Model trains only with at least 80 valid GOOD observations; held-out MAE must beat persistence baseline. No model is trained from static DXF data.")
    st.divider()
    st.subheader("Physics worksheet · sensible heat")
    x,y=st.columns(2)
    with x:
        mdot=st.number_input("Mass flow (kg/s)",min_value=0.0,value=1.0)
        cp=st.number_input("Cp (kJ/kg/K)",min_value=.001,value=2.0)
    with y:
        tin=st.number_input("Inlet temperature (°C)",value=25.0)
        tout=st.number_input("Outlet temperature (°C)",value=150.0)
    st.json(heat_duty(mdot,cp,tin,tout))
    st.caption("These user-entered values are an illustrative physics calculation; no thermodynamic model or Aspen coupling is connected.")

elif page=="Performance cortex":
    st.subheader("Event → sensor → mapped asset → nearby context")
    alert_rows=read_table("alerts")
    if alert_rows.empty:st.info("No approved measurement has breached a configured rule. Import real historian data to activate event analysis.")
    else:
        alert_id=st.selectbox("Inspect event",alert_rows.alert_id.tolist())
        event=alert_rows.set_index("alert_id").loc[alert_id].to_dict()
        ctx=asset_context(event["asset_id"])
        st.warning(f"{ctx['asset']['tag']}: {event['rule']} rule, reading {event['measured']} versus limit {event['threshold']}")
        st.json(advise(event))
        l,r=st.columns(2)
        with l:
            st.markdown("**Evidence chain**")
            st.write({"alert_id":alert_id,"event_id":event["event_id"],"asset_id":event["asset_id"],
                "provenance":ctx["asset"]["provenance"],"nearby_candidates":len(ctx["relationships"])})
            st.dataframe(pd.DataFrame(ctx["relationships"]),hide_index=True,use_container_width=True)
        with r:
            st.markdown("**Measured trend**")
            for mapping in ctx["mappings"]:
                samples=series(mapping["sensor_tag"])
                if len(samples):st.line_chart(samples.set_index("timestamp").value)
    st.caption("This view explains recorded rule breaches and traceable evidence. It does not calculate root cause or causal attribution from drawing proximity.")

elif page=="Model registry":
    st.subheader("Model development and monitoring")
    st.caption("Workflow: load approved measured data → train → compare chronological holdout → record metrics → engineer gate → monitor drift/retrain.")
    path=Path(__file__).resolve().parent/"runtime/models"
    reports=[]
    for entry in sorted(path.glob("*.json")) if path.exists() else []:
        try:reports.append(json.loads(entry.read_text()))
        except ValueError:pass
    if reports:st.dataframe(pd.DataFrame(reports),hide_index=True,use_container_width=True)
    else:st.info("Model registry empty. No predictive model was trained from the DXF.")
    st.markdown("**Vision model:** awaiting engineer-labelled symbol bounding boxes and an untouched drawing-level test set. `validate_yolov5_dataset.py` and `train_yolov5.py` enforce the local training gate; the Azure ML v2 sample is the governed cloud alternative.")
    st.markdown("**Deployment gate:** compare test MAE with persistence, approve per sensor, monitor drift, then register a version. No Kubernetes or production scheduler is configured in this POC.")

elif page=="Alerts & workflows":
    st.subheader("Engineering rule evaluation")
    if st.button("Evaluate approved limits against GOOD readings"):
        st.success(f"Created {evaluate_rules()} new alerts")
    alerts=read_table("alerts")
    if alerts.empty:st.info("No live alerts. Import approved and good-quality readings with breached thresholds to generate alerts.")
    else:
        st.dataframe(alerts,hide_index=True,use_container_width=True)
        row=alerts.iloc[0].to_dict();st.write("**Advisory next step**");st.json(advise(row))
    st.subheader("Drawing validation task queue")
    queue=pd.read_csv(DATA/"validation_queue.csv").fillna("")
    edited=st.data_editor(queue,hide_index=True,use_container_width=True,height=450,
        disabled=["asset_id","tag","asset_class","overall_confidence","evidence_coverage","digital_twin_readiness","review_status"])
    st.download_button("Download engineer decisions",edited.to_csv(index=False).encode(),"validation_queue_reviewed.csv","text/csv")
    if review_unlocked:
        with st.form("review_asset"):
            target=st.selectbox("Asset to review",assets.asset_id.tolist())
            symbol=st.selectbox("Symbol decision",["PENDING","APPROVED","REJECTED"])
            topology=st.selectbox("Connectivity decision",["PENDING","APPROVED","REJECTED"])
            reviewer=st.text_input("Reviewer name",key="asset_reviewer")
            note=st.text_area("Evidence / comment")
            if st.form_submit_button("Record decision"):
                try:record_review(target,symbol,topology,reviewer,note);st.success("Decision and audit event recorded")
                except ValueError as exc:st.error(str(exc))
        st.dataframe(read_table("decisions"),hide_index=True,use_container_width=True)
    else:st.caption("Approval actions require the server-side PID_REVIEW_TOKEN and an identified engineering reviewer.")

elif page=="Copilot & API":
    st.subheader("Evidence-backed asset assistant")
    aid=st.selectbox("Ask about an asset",assets.asset_id.tolist(),format_func=lambda a:f"{a} · {assets.set_index('asset_id').loc[a,'tag']}")
    st.json(summarize_asset(aid))
    st.caption("The assistant constructs a response from database facts and source IDs. It does not call an LLM or invent operating history.")
    with st.expander("Optional approved generative model"):
        st.caption("This sends selected asset context to a configured external endpoint. Use only with company approval for this drawing's data classification.")
        question=st.text_input("Question about selected asset")
        if st.button("Ask configured model"):
            try:
                from llm_adapter import answer
                st.json(answer(aid,question))
            except Exception as exc:st.error(f"LLM adapter error: {exc}")
    st.subheader("REST interface")
    st.code("PID_API_KEY=<your-secret> uvicorn api:app --host 127.0.0.1 --port 8000\nGET /assets, /assets/{asset_id}, /graph, /namespace, /alerts, /copilot/{asset_id}\nPOST /observations · Header: X-API-Key: <your-secret>",language="text")
    st.caption("API documentation becomes available at /docs when the optional API process is running. Use HTTPS, an API gateway and IT-approved storage for production.")

else:
    st.subheader("Data lineage and audit")
    st.json(summary)
    st.dataframe(read_table("audit"),hide_index=True,use_container_width=True)
    st.download_button("Asset register CSV",assets.to_csv(index=False).encode(),"asset_master.csv")
    st.download_button("Candidate edges CSV",rels.to_csv(index=False).encode(),"relationships.csv")
    st.download_button("Knowledge graph JSON",(DATA/"knowledge_graph.json").read_bytes(),"knowledge_graph.json")
    st.caption("Source drawing and original input files are not modified by this app.")
