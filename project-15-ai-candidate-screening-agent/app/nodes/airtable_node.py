from app.services.airtable_client import create_record
from app.states.screening_state import ScreeningState


class AirtableNode:

    def save_candidates(
        self,
        state: ScreeningState
    ):

        record_ids = []

        for candidate in state["scores"]:

            record_id = create_record(
                {
                    "Name": candidate.name,
                    "Score": candidate.score,
                    "Matched Skills": ", ".join(candidate.matched_skills),
                    "Missing Skills": ", ".join(candidate.missing_skills),
                    "Reasoning": candidate.reasoning,
                    "Recommend Outreach": candidate.recommend_outreach
                }
            )

            record_ids.append(record_id)

        return {
            "airtable_record_ids": record_ids
        }