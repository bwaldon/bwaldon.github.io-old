// RESPONSE MODEL, ASSUMING THEMATIC RELATIONS ARE BINARY FEATURES 

var kill_binary = {"arg1" : {proto_agent: ["causation"],
                      proto_patient: []},
            "arg2" : {proto_agent: [],
                      proto_patient: ["causally_affected", "change_of_state"]}}

var murder_binary = {"arg1" : {proto_agent: ["causation"],
                      proto_patient: []},
            "arg2" : {proto_agent: [],
                      proto_patient: ["causally_affected", "change_of_state"]}}

var noise = 0.5

var binary_interpretation = function(arg, proto_role) {
  if ((arg.proto_agent).includes(proto_role) | 
      (arg.proto_patient).includes(proto_role)) {
    return flip(noise) ? "very likely" : "somewhat likely"
  } else {
    return uniformDraw(["very unlikely", "somewhat unlikely", "not enough information"])
  }
}

// RESPONSE MODEL, ASSUMING THEMATIC RELATIONS ARE UNDERLYINGLY BETA DISTRIBUTIONS

var kill_gradient = {"arg1" : {"causation": {type : "proto_agent", a : 5, b : 2},
                               "volition" : {type : "proto_agent", a : 2, b : 5},
                               "sentience" : { type : "proto_agent", a : 2, b : 5},
                               "motion" : {type : "proto_agent", a : 2, b : 5},
                               "change_of_state": {type : "proto_patient", a : 2, b : 5},
                               "incremental_theme" : {type : "proto_patient", a : 2, b : 5},
                               "causally_affected" : {type : "proto_patient", a : 2, b : 5},
                               "stationary" : {type : "proto_patient", a : 2, b : 5}
                              },
                     "arg2" : {"causation": {type : "proto_agent", a : 2, b : 5},
                               "volition" : {type : "proto_agent", a : 2, b : 5},
                               "sentience" : { type : "proto_agent", a : 2, b : 5},
                               "motion" : {type : "proto_agent", a : 2, b : 5},
                               "change_of_state": {type : "proto_patient", a : 5, b : 2},
                               "incremental_theme" : {type : "proto_patient", a : 2, b : 5},
                               "causally_affected" : {type : "proto_patient", a : 5, b : 2},
                               "stationary" : {type : "proto_patient", a : 2, b : 5}
                              }
                    }

var murder_gradient = {"arg1" : {"causation": {type : "proto_agent", a : 10, b : 2},
                               "volition" : {type : "proto_agent", a : 5, b : 2},
                               "sentience" : { type : "proto_agent", a : 5, b : 2},
                               "motion" : {type : "proto_agent", a : 2, b : 5},
                               "change_of_state": {type : "proto_patient", a : 2, b : 5},
                               "incremental_theme" : {type : "proto_patient", a : 2, b : 5},
                               "causally_affected" : {type : "proto_patient", a : 2, b : 5},
                               "stationary" : {type : "proto_patient", a : 2, b : 5}
                              },
                     "arg2" : {"causation": {type : "proto_agent", a : 2, b : 5},
                               "volition" : {type : "proto_agent", a : 2, b : 5},
                               "sentience" : { type : "proto_agent", a : 2, b : 5},
                               "motion" : {type : "proto_agent", a : 2, b : 5},
                               "change_of_state": {type : "proto_patient", a : 5, b : 2},
                               "incremental_theme" : {type : "proto_patient", a : 2, b : 5},
                               "causally_affected" : {type : "proto_patient", a : 5, b : 2},
                               "stationary" : {type : "proto_patient", a : 2, b : 5}
                              }
                    }  
               
var gradient_interpretation = function(arg, proto_role, thetas) {
  var  i = arg[proto_role]
  var p = sample(Beta({a : i.a, b : i.b})) 
  return p > thetas[1] ? "very likely" : 
    p > thetas[2] ? "somewhat likely" : 
    p > thetas[3] ? "not enough information" :
    p > thetas[4] ? "somewhat unlikely" :
    "very unlikely"
  
}


print("Gradient interpretation: how likely did subject arg of 'kill' cause action?")

viz(Infer({ model : function() {
  return gradient_interpretation(kill_gradient.arg1, "causation", [0.8,0.6,0.4,0.2])
}, method : "forward", samples : 100}))

print("Binary interpretation: how likely did subject arg of 'kill' cause action?")

viz(Infer(function() { 
  return binary_interpretation(kill_binary.arg1, "causation") } ))

print("Gradient interpretation: how likely did subject arg of 'murder' cause action?")

viz(Infer({ model : function() {
  return gradient_interpretation(murder_gradient.arg1, "causation", [0.8,0.6,0.4,0.2])
}, method : "forward", samples : 100}))

print("Binary interpretation: how likely did subject arg of 'murder' cause action?")

viz(Infer(function() { 
  return binary_interpretation(murder_binary.arg1, "causation") } ))
