from flask import Flask, request, jsonify
from typing import Any
import vertexai
from vertexai.generative_models import GenerativeModel, Image
from vertexai.preview.vision_models import ImageGenerationModel
import os
import re
import json

app=Flask(__name__)

def initialize_vertex():
    project_id="turito-questions"
    # project_id='turito-ai'
    credentials_path=r"turito-questions-4801fdc9d428.json"
    # credentials_path= "turito-ai-5e5a860a476a.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    vertexai.init(
        project=project_id,
        location="us-central1"
    )
    gemini_model=GenerativeModel("gemini-2.5-flash")

    vertexai.init(
        project=project_id,
        location="us-central1"
    )
    # imagen_model=ImageGenerationModel.from_pretrained("imagen-4.0-ultra-generate-001")
    imagen_model=GenerativeModel("gemini-2.5-flash-image")

    return gemini_model, imagen_model



def get_prompt(subject,grade,chapter,topics,number):
    prompt=f"""
    you are a {subject} education expert creating visual multiple-choice questions for grade: {grade} from the chapter: {chapter} 
    in the topics: {topics}. Generate {number} questions, where EITHER the question or the options or BOTH can be images.
    for each question, specify:
1. **Question Text**: A clear, specific question
2. **Question Image Description**: Description of the main image (question image) (if applicable, otherwise "N/A")
3. **Option Type**: Specify "text" or "image"
4. **Options**: 
   - If text options: Provide 4 text choices (A, B, C, D)
   - If image options: Provide 4 detailed image descriptions (A, B, C, D)
5. **Correct Answer**: The letter of the correct option with brief explanation
6. **Difficulty Level**: Easy/Medium/Hard
7. **Concepts Tested**: List of the Topics covered

Example output format:
{{
    "questions":
[
    {{
        "question_num": 1,
        "question_text": "question text",
        "question_image_description": "detailed image description of the question image",
        "option_type": "text" or "image",
        "options": {{
            "A": "option A",
            "B": "option B",
            "C": "option C",
            "D": "option D"
        }},
        "correct_answer": "correct answer",
        "explanation": "brief explanation",
        "difficulty": "Easy/Medium/Hard",
        "concepts": "Topics covered"
    }},
    {{
        "question_num": 2,
        "question_text": "question text",
        "question_image_description": "null",
        "option_type": "image",
        "options": {{
            "A": "detailed image description of the option A",
            "B": "detailed image description of the option B",
            "C": "detailed image description of the option C",
            "D": "detailed image description of the option D"
        }},
        "correct_answer": "correct answer",
        "explanation": "brief explanation",
        "difficulty": "Easy/Medium/Hard",
        "concepts": "Topics covered"
    }},
    {{
        "question_num": 3,
        "question_text": "question text",
        "question_image_description": "detailed image description of the question image",
        "option_type": "image",
        "options": {{
            "A": "detailed image description of the option A",
            "B": "detailed image description of the option B",
            "C": "detailed image description of the option C",
            "D": "detailed image description of the option D"
        }},
        "correct_answer": "correct answer",
        "explanation": "brief explanation",
        "difficulty": "Easy/Medium/Hard",
        "concepts": "Topics covered"
    }},
]
}}
Do NOT include any general headings like “Visual Multiple-Choice Questions on {subject}” before the questions. 
return the questions STRICTLY in the specified format only.
    """
    return prompt

generation_config = {
    "temperature": 0,
}

def generate_question(model,subject,grade,chapter,topics,number,output_dir):
    prompt_text=get_prompt(subject,grade,chapter,topics,number)
    response=model.generate_content(prompt_text,generation_config=generation_config)
    usage_metadata=response.usage_metadata
    if usage_metadata:
        usage_metadata_dict={
            "prompt_token_count":usage_metadata.prompt_token_count,
            "candidates_token_count":usage_metadata.candidates_token_count,
            "prompt_token_details":usage_metadata.prompt_tokens_details[0].modality,
            "candidate_token_details":usage_metadata.candidates_tokens_details[0].modality

        }
        with open("usage_metadata_text_generation.json",'w',encoding='utf-8') as f:
            json.dump(usage_metadata_dict,f,indent=4)
            
    # with open("question_output.txt","w",encoding="utf-8") as f:
    #     f.write(response.text)
    result_text=""
    result_text+=response.text
    
    if "```json" in result_text:
        json_text=result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        json_text=result_text.split("```")[1].split("```")[0].strip()
    elif result_text.strip().startswith("json"):
        json_text=result_text.strip()[4:].strip()
    else:
        json_text=result_text.strip()
    try:
        json_data=json.loads(json_text)
        with open(f"{output_dir}/question_output.json","w",encoding="utf-8") as f:
            json.dump(json_data,f,indent=4)
        return json_data
    except json.JSONDecodeError as json_err:
        print(f"error in parsing json: {json_err}")
        with open("question_output.txt","w",encoding="utf-8") as f:
            f.write(result_text)
        return {
            "questions" :[]
        }   



def create_composite_question_card(question_data,imagen_model,output_path,subject):
    base_style="""
    Style requirements:
    - Clean, textbook-quality scientific illustration
    - Clear, readable labels using Arial or similar sans-serif font
    - Pure white background
    - Professional educational style suitable for School
    - High contrast for easy viewing
    - Simple and uncluttered design
    - Ensure all text and symbols are clearly legible
    - No watermarks or additional text
"""
    prompt_parts=[]
    prompt_parts.append(f"you are an expert in creating educational question cards. you will be given image descriptions of the question and the options. Understand the question description and the option descriptions properly and create a complete educational {subject} question card with the following layout:")
    prompt_parts.append("")
    if question_data.get('question_image_description'):
        prompt_parts.append("TOP SECTION:")
        prompt_parts.append(f"- {question_data['question_image_description']}")
        prompt_parts.append("")
    prompt_parts.append("QUESTION TEXT (centered, bold, 18pt font):")
    prompt_parts.append(f'"{question_data["question_text"]}"')
    prompt_parts.append("")
    prompt_parts.append("OPTIONS SECTION (arranged in 2x2 grid below the question):")
    if question_data['option_type'] == 'text':
        for letter in ['A', 'B', 'C', 'D']:
            option_text = question_data['options'].get(letter, '')
            prompt_parts.append(f"  {letter}) {option_text}")
    else:
        prompt_parts.append("Display the following 4 diagrams in a 2x2 grid layout:")
        for letter in ['A', 'B', 'C', 'D']:
            option_desc = question_data['options'].get(letter, '')
            prompt_parts.append(f"  Option {letter} (labeled clearly): {option_desc}")
    prompt_parts.append("")
    prompt_parts.append("LAYOUT REQUIREMENTS:")
    prompt_parts.append("- Professional educational poster format")
    prompt_parts.append("- Clean white background")
    prompt_parts.append("- Clear section separations with subtle lines")
    prompt_parts.append("- Option labels (A, B, C, D) in bold circles")
    prompt_parts.append("- Consistent spacing and alignment")
    prompt_parts.append("- All elements clearly visible and readable")
    prompt_parts.append("- High quality, suitable for printing")

    full_prompt=base_style+"\n".join(prompt_parts)
    try:
        print("Generating composite question card...")
        # response=imagen_model.generate_images(
        #     prompt=full_prompt,
        #     number_of_images=1,
        #     aspect_ratio="3:4",
        #     safety_filter_level="block_some",
        #     person_generation="allow_adult"
        # )
        # if response.images and len(response.images) > 0:
        #     response.images[0].save(output_path)
        #     print(f"Composite card saved: {os.path.basename(output_path)}")
        #     return output_path
        # else:
        #     print(f"Failed to generate composite card")
        #     return None

        response=imagen_model.generate_content(full_prompt)
        usage_metadata=response.usage_metadata
        print(usage_metadata)
        if usage_metadata:
            usage_metadata_dict={
            "prompt_token_count":usage_metadata.prompt_token_count,
            "candidates_token_count":usage_metadata.candidates_token_count,
            "prompt_token_details":usage_metadata.prompt_tokens_details[0].modality,
            "candidate_token_details":usage_metadata.candidates_tokens_details[0].modality

        }
            with open("usage_metadata_image_generation_nano_banana.json",'w',encoding='utf-8') as f:
                json.dump(usage_metadata_dict,f,indent=4)
        image_count=0
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part,'mime_type') and part.inline_data:
                    image_data=part.inline_data.data
                    mime_to_ext={
                        "image/png":"png",
                        "image/jpeg":"jpg",
                        "image/webp":"webp"
                    }

                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    
                    image_count+=1
            
        if image_count == 0:
            print("\n No images found in response")
            print("The model may have returned text instead of generating an image")
        return output_path

    except Exception as e:
        print(f"Error generating composite card: {str(e)}")
        return None

def process_question_with_scenarios(question,imagen_model, output_dir,subject,scenario='composite'):
    question_num=question['question_num']

    question_data={
        'question_num':question_num,
        'question_text':question.get('question_text',''),
        'question_image_description':question.get('question_image_description'),
        'option_type':question.get('option_type', 'text'),
        'options':question.get('options',{}),
        'correct_answer': question.get('correct_answer',''),
        'explanation':question.get('explanation',''),
        'difficulty':question.get('difficulty',''),
        'concepts':question.get('concepts',''),
        'image_files':{}
    }

    # q_dir = os.path.join(output_dir,f'question_{question_num}')
    q_dir = output_dir
    os.makedirs(q_dir,exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing Question {question_num} (Scenario: {scenario})")
    print(f"{'='*60}")
    print(f"Question: {question_data['question_text'][:80]}...")
    print(f"Option Type: {question_data['option_type']}")

    if scenario == 'composite':
        print("\n📦 Generating composite question card...")
        composite_filename=f"Q{question_num}_complete_card.png"
        composite_path=os.path.join(q_dir,composite_filename)

        result=create_composite_question_card(question_data,imagen_model,composite_path,subject)

        if result:
            question_data['image_files']['composite_card']=composite_filename
        
    return question_data

    

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    data=request.get_json()
    subject=data.get('subject')
    grade=data.get('grade')
    chapter=data.get('chapter')
    topics=data.get('topics')
    number=data.get('number')
    gemini_model,imagen_model= initialize_vertex()
    output_dir=f"{subject}_{grade}_{chapter}_nano_banana_v2"
    os.makedirs(output_dir,exist_ok=True)
    gemini_response=generate_question(gemini_model,subject,grade,chapter,topics,number,output_dir)
    questions= gemini_response.get('questions', [])

    

    processed_question=[]

    for question in questions:
        processed=process_question_with_scenarios(question,imagen_model,output_dir,subject,scenario='composite')
        processed_question.append(processed)

    return jsonify({
        'status': 'success',
        'message': 'Questions generated successfully',
        'output_dir': output_dir,
        'question_count': len(processed_question),
        'questions': processed_question
    }),200

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=False, port=5000)








    

    
