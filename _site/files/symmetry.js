// PREDICTING SYMMETRY INTUITIONS (ASSUME: PARAMETERS OF DISTRIBUTIONS INFERRED EMPIRICALLY)

var divorce_gradient = {"arg1" : {"causation": {type : "proto_agent", a : 20, b : 2},
                               "volition" : {type : "proto_agent", a : 20, b : 2},
                               "sentience" : { type : "proto_agent", a : 20, b : 2},
                               "motion" : {type : "proto_agent", a : 2, b : 20},
                               "change_of_state": {type : "proto_patient", a : 2, b : 20},
                               "incremental_theme" : {type : "proto_patient", a : 2, b : 20},
                               "causally_affected" : {type : "proto_patient", a : 2, b : 20},
                               "stationary" : {type : "proto_patient", a : 2, b : 20}
                              },
                        // OBJECT ARGUMENT HAS LESS VOLITION THAN SUBJECT ARGUMENT
                     "arg2" : {"causation": {type : "proto_agent", a : 20, b : 2},
                               "volition" : {type : "proto_agent", a : 5, b : 2},
                               "sentience" : { type : "proto_agent", a : 20, b : 2},
                               "motion" : {type : "proto_agent", a : 2, b : 20},
                               "change_of_state": {type : "proto_patient", a : 2, b : 20},
                               "incremental_theme" : {type : "proto_patient", a : 2, b : 20},
                               "causally_affected" : {type : "proto_patient", a : 2, b : 20},
                               "stationary" : {type : "proto_patient", a : 2, b : 20}
                              }
                    }

var roles = ["causation","volition","sentience","motion","change_of_state",
            "incremental_theme","causally_affected","stationary"]

var is_symmetric = function (predicate) {
 var arg1_protoroles = map(function (role) {
    var i = predicate.arg1[role]
    var p = sample(Beta({a: i.a, b: i.b}))
    flip(p) ? true : false 
  }, roles)
 var arg2_protoroles = map(function (role) {
    var i = predicate.arg2[role]
    var p = sample(Beta({a: i.a, b: i.b}))
    flip(p) ? true : false
  }, roles) 
_.isEqual(arg1_protoroles,arg2_protoroles) ? "symmetric" : "asymmetric"
  
}
                    
// DUMMY THRESHOLDS

Infer({model : function () { is_symmetric(divorce_gradient) }, 
       method : "forward", samples: 1000})
