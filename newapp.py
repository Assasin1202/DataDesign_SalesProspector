import streamlit as st
import pandas as pd
from new_backend import process_lead_data, ModelType, AzureModelType
import json

def main():
    # Set page config must be the first Streamlit command
    st.set_page_config(
        page_title="Researcher Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for styling
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        div[data-testid="stFileUploader"] {
            border: 2px dashed #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            background-color: white;
        }
        div[data-testid="stFileUploader"] section {
            gap: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and subtitle in a centered container
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='margin-bottom: 1rem;'>Researcher Agent</h1>
            <p style='color: #666; font-size: 1.1rem;'>Upload your lead data to view and analyze</p>
        </div>
    """, unsafe_allow_html=True)

    # --- UPLOAD SECTION ---
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container():
            uploaded_file = st.file_uploader(
                "Upload your XLSX lead data file",
                type=['xlsx'],
                label_visibility="collapsed"
            )

    # --- DATA DISPLAY SECTION ---
    if uploaded_file:
        st.markdown("---")
        
        # File info with icon
        st.markdown(f"""
            <div style='display: flex; align-items: center; margin-bottom: 1rem; font-size: 1.1rem;'>
                <span style='font-size: 1.5rem; margin-right: 0.5rem;'>📄</span>
                <div>
                    <strong>{uploaded_file.name}</strong>
                    <small style='color: #666; margin-left: 10px;'>({uploaded_file.size/1024:.1f} KB)</small>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container(border=True):
            try:
                # Read the Excel file
                df = pd.read_excel(uploaded_file)
                
                # Show the dataframe
                st.markdown("### Data Preview")
                st.dataframe(df, use_container_width=True)
                
                # Show basic statistics in a styled container
                st.markdown("### 📊 Basic Statistics")
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Total Records", len(df))
                with stat_col2:
                    st.metric("Total Columns", len(df.columns))
                
                st.markdown("#### Column Names")
                st.markdown(", ".join([f"`{col}`" for col in df.columns]))
                
            except Exception as e:
                st.error(f"Error reading the Excel file: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

        # --- MODEL SELECTION ---
        st.markdown("---")
        st.markdown("### ⚙️ Model Selection")
        provider_display = {
            ModelType.AZURE: "Azure GPT (OpenAI)",
            ModelType.MISTRAL: "Mistral",
            ModelType.CLAUDE: "Claude"
        }
        provider_option = st.selectbox(
            "Choose Model Provider",
            options=list(provider_display.keys()),
            format_func=lambda x: provider_display[x]
        )
        azure_model_option = None
        if provider_option == ModelType.AZURE:
            azure_model_display = {
                AzureModelType.GPT4_1: "GPT-4.1 (32k)",
                AzureModelType.GPT4_1_MINI: "GPT-4.1 Mini",
                AzureModelType.O3_MINI: "O3 Mini",
                AzureModelType.O4_MINI: "O4 Mini"
            }
            azure_model_option = st.selectbox(
                "Choose Azure Deployment",
                options=list(azure_model_display.keys()),
                format_func=lambda x: azure_model_display[x]
            )
        else:
            azure_model_option = AzureModelType.GPT4_1_MINI  # default, unused

        # --- ENRICHMENT & EVALUATION SECTION ---
        st.markdown("---")
        st.markdown("### 🔍 Enrich & Evaluate Lead")
        st.markdown("#### Preferred Outreach Channel (optional)")
        channel_option = st.selectbox(
            "Choose preferred channel",
            options=["Auto", "Email", "LinkedIn"],
            index=0,
            format_func=lambda x: "Let the agent decide" if x == "Auto" else x
        )

        # --- UNIFIED MESSAGING PREFERENCES ---
        st.markdown("### 💬 Messaging Preferences")
        unified_preferences = st.text_area(
            "Messaging Agent Preferences (optional)",
            placeholder="Example: Keep tone formal, mention 24/7 support. Company: TechSolutions Inc - software development services",
            help="Unified input for all messaging customizations. Include messaging preferences, company info, or both.",
            height=120,
        )

        # --- CUSTOM GRADING PROMPT (SIMPLE) ---
        with st.expander("🎯 Custom Grading Prompt (optional)"):
            custom_prompt_text = st.text_area(
                "Paste or write your own grading prompt here to override the default (include ICP, assessment factors, scoring criteria, etc.).",
                placeholder="Enter custom prospect grading prompt...",
                height=250,
            )

        if st.button("Run Enrichment & Evaluation"):
            with st.spinner("Running enrichment and evaluation on the first row..."):
                try:
                    preferred_channel = None if channel_option == "Auto" else channel_option

                    # Decide which custom prompt to send: advanced builder takes precedence over legacy text area
                    custom_prompt_to_send = None
                    if 'custom_prompt_text' in locals() and custom_prompt_text:
                        custom_prompt_to_send = custom_prompt_text

                    initial_lead, enriched_lead, evaluation_result, grading_result, messaging_result = process_lead_data(
                        uploaded_file.getvalue(),
                        model_type=provider_option,
                        azure_model=azure_model_option,
                        preferred_channel=preferred_channel,
                        unified_messaging_preferences=unified_preferences.strip() if unified_preferences else None,
                        custom_grading_prompt=custom_prompt_to_send,
                    )
                    if enriched_lead is None or evaluation_result is None:
                        st.error("Enrichment or evaluation failed. Please check your data and try again.")
                    else:
                        st.success("✅ Enrichment and evaluation complete!")
                        # Show enriched paragraph
                        st.markdown("#### Enriched Paragraph")
                        st.info(enriched_lead.enriched_paragraph)
                        # Show web sources
                        st.markdown("#### Web Sources Used")
                        for url in enriched_lead.web_sources:
                            st.markdown(f"- [{url}]({url})")
                        # Show evaluation results
                        st.markdown("#### Evaluation Results")
                        st.metric("Factual Accuracy Score", evaluation_result.factual_accuracy_score)
                        st.metric("Temporal Accuracy Score", evaluation_result.temporal_accuracy_score)
                        st.metric("Source Citation Score", evaluation_result.source_citation_score)
                        st.markdown("**Evaluation Summary:**")
                        st.write(evaluation_result.evaluation_summary)
                        st.markdown("**Identified Discrepancies:**")
                        if evaluation_result.identified_discrepancies:
                            for d in evaluation_result.identified_discrepancies:
                                st.warning(d)
                        else:
                            st.success("No discrepancies found.")

                        # Show grading results
                        if grading_result:
                            st.markdown("#### Prospect Grading")

                            # Display dynamic category scores in a table
                            scores_data = [
                                {
                                    "Category": entry.category,
                                    "Score (1-5)": entry.score,
                                    "Reason": entry.reason,
                                }
                                for entry in grading_result.scores
                            ]
                            st.table(pd.DataFrame(scores_data))

                            # Summary metrics
                            metric_cols = st.columns(2)
                            if grading_result.total_score is not None:
                                metric_cols[0].metric("Total Score", grading_result.total_score)
                            metric_cols[1].metric("Priority", grading_result.priority_service)

                            st.markdown("**Key Insights:**")
                            for bullet in grading_result.key_insights:
                                st.write(f"- {bullet}")

                        # Show personalized message
                        if messaging_result:
                            st.markdown("#### Personalized Outreach Message")
                            st.write(f"**Channel:** {messaging_result.channel}")
                            st.write(f"**Subject:** {messaging_result.subject}")
                            st.info(messaging_result.message)

                        # Optionally, show JSON
                        with st.expander("Show Raw Output (JSON)"):
                            st.json({
                                "enriched_lead": json.loads(enriched_lead.model_dump_json()),
                                "evaluation_result": json.loads(evaluation_result.model_dump_json()),
                                "grading_result": json.loads(grading_result.model_dump_json()) if grading_result else {},
                                "messaging_result": json.loads(messaging_result.model_dump_json()) if messaging_result else {}
                            })
                except Exception as e:
                    st.error(f"Error during enrichment/evaluation: {str(e)}")

    # Footer
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; color: #666; font-family: sans-serif;'>
            Made by Pranav Pant. 
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()