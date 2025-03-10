from flask import Flask, request, jsonify
from ortools.sat.python import cp_model

app = Flask(__name__)

def solve_scheduling(data):
    model = cp_model.CpModel()
    
    # Paramètres globaux
    D = data.get("days", 3)            # Nombre de jours
    H = data.get("slots_per_day", 10)    # Nombre de créneaux par jour
    margin = data.get("margin", 1)       # Marge entre examens (en créneaux)
    
    exams = data.get("exams", [])
    rooms = data.get("rooms", [])
    
    num_exams = len(exams)
    
    # Variables de décision pour chaque examen
    exam_start = {}  # instant global (0 à D*H - durée)
    exam_day = {}    # jour de l'examen (0 à D-1)
    exam_slot = {}   # créneau dans le jour (0 à H-1)
    exam_room = {}   # salle attribuée (indice dans la liste rooms)
    
    for e, exam in enumerate(exams):
        duration = exam.get("duration", 1)
        # L'examen doit débuter suffisamment tôt pour tenir dans l'horizon
        exam_start[e] = model.NewIntVar(0, D * H - duration, f'start_{e}')
        # On dérive le jour et le créneau à partir de exam_start
        exam_day[e] = model.NewIntVar(0, D - 1, f'day_{e}')
        exam_slot[e] = model.NewIntVar(0, H - 1, f'slot_{e}')
        model.AddDivisionEquality(exam_day[e], exam_start[e], H)
        model.AddModuloEquality(exam_slot[e], exam_start[e], H)
        # L'examen ne doit pas dépasser le dernier créneau du jour
        model.Add(exam_slot[e] + duration <= H)
        
        # Affectation de salle basée sur la capacité
        students = exam.get("students", 0)
        possible_rooms = [r for r, room in enumerate(rooms) if room.get("capacity", 0) >= students]
        exam_room[e] = model.NewIntVarFromDomain(cp_model.Domain.FromValues(possible_rooms), f'room_{e}')
    
    # Contraintes de non-chevauchement dans la même salle ET le même jour
    for i in range(num_exams):
        for j in range(i + 1, num_exams):
            same_room = model.NewBoolVar(f'same_room_{i}_{j}')
            model.Add(exam_room[i] == exam_room[j]).OnlyEnforceIf(same_room)
            model.Add(exam_room[i] != exam_room[j]).OnlyEnforceIf(same_room.Not())
            
            same_day = model.NewBoolVar(f'same_day_{i}_{j}')
            model.Add(exam_day[i] == exam_day[j]).OnlyEnforceIf(same_day)
            model.Add(exam_day[i] != exam_day[j]).OnlyEnforceIf(same_day.Not())
            
            # Si les examens se déroulent dans la même salle et le même jour, ils ne doivent pas se chevaucher
            both = model.NewBoolVar(f'both_{i}_{j}')
            model.AddBoolAnd([same_room, same_day]).OnlyEnforceIf(both)
            model.AddBoolOr([same_room.Not(), same_day.Not()]).OnlyEnforceIf(both.Not())
            
            duration_i = exams[i].get("duration", 1)
            duration_j = exams[j].get("duration", 1)
            i_before_j = model.NewBoolVar(f'i_before_j_{i}_{j}')
            j_before_i = model.NewBoolVar(f'j_before_i_{i}_{j}')
            model.Add(exam_slot[i] + duration_i + margin <= exam_slot[j]).OnlyEnforceIf(i_before_j)
            model.Add(exam_slot[j] + duration_j + margin <= exam_slot[i]).OnlyEnforceIf(j_before_i)
            model.AddBoolOr([i_before_j, j_before_i]).OnlyEnforceIf(both)
    
    # Contraintes pour les promotions : deux examens de promotions différentes ne peuvent pas se dérouler simultanément le même jour.
    for i in range(num_exams):
        for j in range(i + 1, num_exams):
            promo_i = exams[i].get("promotion")
            promo_j = exams[j].get("promotion")
            if promo_i is not None and promo_j is not None and promo_i != promo_j:
                same_day_promo = model.NewBoolVar(f'promo_same_day_{i}_{j}')
                model.Add(exam_day[i] == exam_day[j]).OnlyEnforceIf(same_day_promo)
                model.Add(exam_day[i] != exam_day[j]).OnlyEnforceIf(same_day_promo.Not())
                
                duration_i = exams[i].get("duration", 1)
                duration_j = exams[j].get("duration", 1)
                i_before_j = model.NewBoolVar(f'promo_i_before_j_{i}_{j}')
                j_before_i = model.NewBoolVar(f'promo_j_before_i_{i}_{j}')
                model.Add(exam_slot[i] + duration_i <= exam_slot[j]).OnlyEnforceIf(i_before_j)
                model.Add(exam_slot[j] + duration_j <= exam_slot[i]).OnlyEnforceIf(j_before_i)
                model.AddBoolOr([i_before_j, j_before_i]).OnlyEnforceIf(same_day_promo)
    
    # Fonction objectif : minimiser la durée totale de la période d’examen
    exam_end = []
    for e in range(num_exams):
        duration = exams[e].get("duration", 1)
        exam_end_var = model.NewIntVar(0, D * H, f'end_{e}')
        model.Add(exam_end_var == exam_start[e] + duration)
        exam_end.append(exam_end_var)
    
    start_min = model.NewIntVar(0, D * H, 'start_min')
    end_max = model.NewIntVar(0, D * H, 'end_max')
    model.AddMinEquality(start_min, [exam_start[e] for e in range(num_exams)])
    model.AddMaxEquality(end_max, exam_end)
    period = model.NewIntVar(0, D * H, 'period')
    model.Add(period == end_max - start_min)
    model.Minimize(period)
    
    # Résolution
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    results = []
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for e in range(num_exams):
            exam = exams[e]
            result = {
                "name": exam.get("name"),
                "filiere": exam.get("filiere"),  # champ ajouté même s'il n'est pas utilisé dans la modélisation
                "promotion": exam.get("promotion"),
                "day": solver.Value(exam_day[e]),
                "slot": solver.Value(exam_slot[e]),
                "room": rooms[solver.Value(exam_room[e])]["name"]
            }
            results.append(result)
        return {"status": "success", "results": results, "total_period": solver.Value(period)}
    else:
        return {"status": "failure", "message": "Aucune solution trouvée"}

@app.route('/api/schedule', methods=['POST'])
def schedule():
    """
    Endpoint pour recevoir les données au format JSON et renvoyer la programmation des examens.
    """
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Aucune donnée fournie"}), 400
    
    result = solve_scheduling(data)
    return jsonify(result)

@app.route('/api/schedule/format', methods=['GET'])
def schedule_format():
    """
    Endpoint pour renvoyer un exemple du format attendu pour envoyer les données de planification.
    """
    sample_format = {
        "days": 3,
        "slots_per_day": 10,
        "margin": 1,
        "exams": [
            {
                "name": "Mathématiques",
                "duration": 2,
                "students": 30,
                "promotion": 1,
                "filiere": "GL"
            },
            {
                "name": "Physique",
                "duration": 3,
                "students": 25,
                "promotion": 2,
                "filiere": "IA"
            }
        ],
        "rooms": [
            {
                "name": "Salle A",
                "capacity": 35
            },
            {
                "name": "Salle B",
                "capacity": 50
            }
        ]
    }
    return jsonify(sample_format)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
