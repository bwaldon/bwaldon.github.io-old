// priors over speaker beliefs of likelihood of rain

var rain = {
  "likelihoods": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
//   // speaker assumed to have uniform prior beliefs:
//   "probabilities": [1,1,1,1,1,1,1,1,1,1,1]
  // speaker assumed to be highly knowledgable about weather (i.e. certain of its truth or falsity):
  "probabilities": [10,1,1,1,1,1,1,1,1,1,10]
//   // speaker assumed to be maximally certain of rain: 
//   "probabilities": [1,1,1,1,1,1,1,1,1,1,10]
//   // speaker assumed to be maximally certain of ~rain:
//   "probabilities": [10,1,1,1,1,1,1,1,1,1,1]
};

// priors over probability thresholds for 'must'

var mustPrior = {
  "likelihoods": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
  "probabilities": [0.5,1,1,2,2,3,3,4,4,5,5]
  // should: 
//    "probabilities": [1,2,2,3,3,4,4,5,5,4,4]
 };

var thetaPrior = function() {
  return categorical(mustPrior.probabilities, mustPrior.likelihoods);
};

var statePrior = function() {
  return categorical(rain.probabilities, rain.likelihoods);
};

var alpha = 4; // optimality parameter

var utterances = ["must", "bare", ""];
var utterancePrior = function() {
  return uniformDraw(utterances);
};

var cost = {
  "must": 2,
  "bare" : 1,
  "" : 0
};

var meaning = function(utterance, likelihood, theta) {
  utterance == "must" ? likelihood >= theta : utterance == "bare" ? likelihood == 1 : true;
};

var literalListener = cache(function(utterance, theta) {
  return Infer({method: "enumerate"}, function() {
    var likelihood = uniformDraw(rain.likelihoods);
    condition(meaning(utterance, likelihood, theta))
    return likelihood;
  });
});

var speaker = cache(function(likelihood, theta) {
  return Infer({method: "enumerate"}, function() {
    var utterance = utterancePrior();
    factor( alpha * (literalListener(utterance, theta).score(likelihood) 
                    - cost[utterance]));
    return utterance;
  });
});

var pragmaticListener = function(utterance) {
  return Infer({method: "enumerate"}, function() {
    var likelihood = statePrior();
    var theta = thetaPrior();
    factor(speaker(likelihood, theta).score(utterance));
    return { likelihood: likelihood, theta: theta };
  });
};

print('Prior prob that threshold for "must" = 1: ')
print(Math.exp(Infer(thetaPrior).score(1)))

print('Posterior prob that threshold for "must" = 1: ')
var mustRain = pragmaticListener("must");
print(Math.exp(marginalize(mustRain, "theta").score(1)))

print('Prior prob that speaker is certain of rain: ')
print(Math.exp(Infer(statePrior).score(1)))

print('Posterior prob speaker is certain of rain: ')
var mustRain = pragmaticListener("must");
print(Math.exp(marginalize(mustRain, "likelihood").score(1)))

// pragmaticListener2("must")
print("B_phi prior: ")
print(Infer(statePrior))

print("Theta_must prior: ")
print(Infer(thetaPrior))

print("B_phi posterior: ")
print(marginalize(mustRain, "likelihood"))

print("Theta_must posterior: ")
print(marginalize(mustRain, "theta"))
