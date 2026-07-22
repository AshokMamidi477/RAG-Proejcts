from app.states.screening_state import ScreeningState
from app.states.jd_criteria import JDCriteria
from app.states.candidate_score import CandidateScore


class ScreeningNode:

    def __init__(self, llm):
        self.llm = llm


    def node_analyse_jd(self, state: ScreeningState):
     criteria = self.analyse_jd(state["jd_text"])

     return {
        "criteria": criteria
    }


    def analyse_jd(self, jd_text: str) -> JDCriteria:
        """
        Node 1: Extract job description criteria.
        """

        structured_llm = self.llm.with_structured_output(JDCriteria)

        prompt = f"""
        Extract the key hiring criteria from this job description.

        Return:
        - required_skills
        - nice_to_have
        - min_years_experience
        - role_title
        - key_responsibilities

        JOB DESCRIPTION:

        {jd_text}
        """
        return structured_llm.invoke(prompt)

    def node_score_resumes(self, state: ScreeningState):

        scores = []
        for resume in state["resumes"]:
                score = self.score_resume(
                    resume["text"],
                    resume["name"],
                    state["criteria"]
                )

                scores.append(score)

        return {
            "scores": scores
    }

    def score_resume(
        self,
        resume_text: str,
        candidate_name: str,
        criteria: JDCriteria
    ) -> CandidateScore:
        """
        Node 2: Score a single resume against JD criteria.
        """

        structured_llm = self.llm.with_structured_output(
            CandidateScore
        )

        prompt = f"""
        Score this candidate's resume against the job criteria.

        JOB CRITERIA:

        Role:
        {criteria.role_title}

        Required Skills:
        {criteria.required_skills}

        Minimum Experience:
        {criteria.min_years_experience}

        Responsibilities:
        {criteria.key_responsibilities}


        RESUME:

        {resume_text}


        Candidate Name:
        {candidate_name}


        Evaluate:
        - Overall score (0-100)
        - Matched skills
        - Missing skills
        - Reasoning
        - Recommend outreach if score >= 70

        """

        return structured_llm.invoke(prompt)


    def node_draft_outreach(self, state: ScreeningState):
      emails = []
      for candidate in state["scores"]:
        if candidate.recommend_outreach:
                emails.append(
                    self.draft_outreach(
                        candidate,
                        state["criteria"].role_title,
                        state["company"]
                )
            )

      return { "outreach_emails": emails}


    def draft_outreach(
        self,
        candidate: CandidateScore,
        role_title: str,
        company: str = "our company"
    ) -> str:
        """
        Node 3: Generate recruiter outreach email.
        """

        prompt = f"""
        Draft a personalized recruiter outreach email.

        Candidate:
        {candidate.name}

        Role:
        {role_title} at {company}

        Matched Skills:
        {candidate.matched_skills}

        Candidate Score:
        {candidate.score}/100


        Write a 3 paragraph email:

        Paragraph 1:
        Why we noticed this candidate.

        Paragraph 2:
        What makes this role exciting.

        Paragraph 3:
        Clear call to action.

        Tone:
        Warm and professional.

        Length:
        150-200 words.
        """

        response = self.llm.invoke(prompt)

        return response.content