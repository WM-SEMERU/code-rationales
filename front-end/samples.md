time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      2.6769146188598825e-06
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " clamp"
  },
  "10": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      3.611827651184285e-06,
      3.524046860547969e-06,
      2.36320192925632e-06,
      0.00018838804680854082,
      0.0019780280999839306,
      0.0013553306926041842,
      0.00036731691216118634,
      6.888843927299604e-05,
      0.0003957811859436333,
      1.4153983762810185e-08
    ],
    "rationales": [
      "def",
      " clamp",
      "(",
      "x",
      ",",
      " lo",
      ",",
      " hi",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9
    ],
    "token": " "
  },
  "11": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.999504566192627
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      10
    ],
    "token": " "
  },
  "12": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.9994699358940125
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      11
    ],
    "token": " "
  },
  "13": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0023775878362357616,
      0.01669282838702202,
      0.00023129944747779518,
      0.015670157968997955,
      0.019718587398529053,
      0.017332937568426132,
      0.01437391247600317,
      0.0012219821801409125,
      0.0005131859797984362,
      0.009317444637417793,
      0.019260026514530182,
      0.017215829342603683,
      9.001880130199424e-08
    ],
    "rationales": [
      "def",
      " clamp",
      "(",
      "x",
      ",",
      " lo",
      ",",
      " hi",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12
    ],
    "token": " if"
  },
  "14": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.012675972655415535,
      0.059321049600839615,
      0.01623864471912384,
      0.2952673137187958,
      0.00010706898319767788
    ],
    "rationales": [
      "def",
      "(",
      "x",
      "):",
      " if"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      8,
      13
    ],
    "token": " x"
  },
  "15": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Functional"
    ],
    "probabilities": [
      0.0041067092679440975,
      0.1617029905319214,
      0.1325988620519638,
      0.027291174978017807,
      5.1051109039690346e-05
    ],
    "rationales": [
      "def",
      " ",
      " ",
      " if",
      " x"
    ],
    "rationales_indexes": [
      0,
      11,
      12,
      13,
      14
    ],
    "token": " <"
  },
  "16": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.5706115961074829,
      5.087896326472219e-09
    ],
    "rationales": [
      " lo",
      " <"
    ],
    "rationales_indexes": [
      5,
      15
    ],
    "token": " lo"
  },
  "17": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.011293700896203518,
      0.0921410620212555,
      0.0122156897559762,
      5.28766086471677e-10
    ],
    "rationales": [
      "def",
      " hi",
      "):",
      " lo"
    ],
    "rationales_indexes": [
      0,
      7,
      8,
      16
    ],
    "token": ":"
  },
  "18": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.15129072964191437,
      0.6742175817489624,
      0.07367224991321564,
      0.22693710029125214,
      4.96731445309706e-06
    ],
    "rationales": [
      "):",
      "\n",
      " ",
      " if",
      ":"
    ],
    "rationales_indexes": [
      8,
      9,
      10,
      13,
      17
    ],
    "token": "\n"
  },
  "19": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.0006570547702722251,
      0.8780653476715088,
      7.247776068197709e-08
    ],
    "rationales": [
      " ",
      ":",
      "\n"
    ],
    "rationales_indexes": [
      12,
      17,
      18
    ],
    "token": " "
  },
  "2": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.018983466550707817,
      5.6411257887134525e-09
    ],
    "rationales": [
      "def",
      " clamp"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "("
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.999420166015625
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      19
    ],
    "token": " "
  },
  "21": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.999428927898407
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      20
    ],
    "token": " "
  },
  "22": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9994227886199951
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      21
    ],
    "token": " "
  },
  "23": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.999372661113739
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      22
    ],
    "token": " "
  },
  "24": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.9994456171989441
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      23
    ],
    "token": " "
  },
  "25": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.9993972778320312
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      24
    ],
    "token": " "
  },
  "26": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Functional"
    ],
    "probabilities": [
      2.6751280529424548e-05,
      0.03232072666287422,
      0.013371921144425869,
      0.0061335451900959015,
      0.09613275527954102,
      0.0006900042062625289,
      0.0540461502969265,
      5.468363539762322e-10
    ],
    "rationales": [
      "def",
      ",",
      " hi",
      "):",
      " ",
      " if",
      "\n",
      " "
    ],
    "rationales_indexes": [
      0,
      6,
      7,
      8,
      12,
      13,
      18,
      25
    ],
    "token": " return"
  },
  "27": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      9.199096530210227e-05,
      0.007282376289367676,
      0.11506008356809616,
      1.6328229435202957e-08
    ],
    "rationales": [
      "def",
      " lo",
      "):",
      " return"
    ],
    "rationales_indexes": [
      0,
      5,
      8,
      26
    ],
    "token": " lo"
  },
  "28": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.5897477865219116,
      1.024703100149793e-09
    ],
    "rationales": [
      "\n",
      " lo"
    ],
    "rationales_indexes": [
      18,
      27
    ],
    "token": "\n"
  },
  "29": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.057541556656360626,
      0.0059632305055856705,
      0.8363801836967468,
      2.8258369866307476e-07
    ],
    "rationales": [
      " ",
      " ",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      24,
      25,
      26,
      28
    ],
    "token": " "
  },
  "3": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.01792759820818901,
      0.016088707372546196,
      4.3236294004600495e-05
    ],
    "rationales": [
      "def",
      " clamp",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "x"
  },
  "30": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.9993077516555786
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      29
    ],
    "token": " "
  },
  "31": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Signature"
    ],
    "probabilities": [
      0.9993191957473755
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      30
    ],
    "token": " "
  },
  "32": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.0003495067940093577,
      0.023284222930669785,
      0.20162810385227203,
      1.3984578117742785e-07
    ],
    "rationales": [
      "(",
      " if",
      "\n",
      " "
    ],
    "rationales_indexes": [
      2,
      13,
      28,
      31
    ],
    "token": " if"
  },
  "33": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.009918450377881527,
      0.0030939499847590923,
      0.0547371543943882,
      0.09519147872924805,
      0.036508865654468536,
      0.14958727359771729,
      0.00013442497584037483
    ],
    "rationales": [
      "def",
      "x",
      "):",
      " if",
      " x",
      " return",
      " if"
    ],
    "rationales_indexes": [
      0,
      3,
      8,
      13,
      14,
      26,
      32
    ],
    "token": " x"
  },
  "34": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.0016230791807174683,
      0.04377862438559532,
      0.033579882234334946,
      0.08731846511363983,
      0.015056919306516647,
      0.05120113864541054,
      2.545054478275688e-08
    ],
    "rationales": [
      "def",
      " lo",
      " hi",
      " if",
      " <",
      " lo",
      " x"
    ],
    "rationales_indexes": [
      0,
      5,
      7,
      13,
      15,
      16,
      33
    ],
    "token": " >"
  },
  "35": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Bool"
    ],
    "probabilities": [
      5.989823694108054e-05,
      0.04575878009200096,
      0.0016108120325952768,
      0.10621485114097595,
      0.1457388699054718,
      0.01379523053765297,
      1.1266104138485389e-06
    ],
    "rationales": [
      "def",
      "x",
      " hi",
      "):",
      "\n",
      " if",
      " >"
    ],
    "rationales_indexes": [
      0,
      3,
      7,
      8,
      28,
      32,
      34
    ],
    "token": " hi"
  },
  "36": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.03782622516155243,
      0.03198177367448807,
      0.0018575561698526144,
      0.009532911702990532,
      0.012013963423669338,
      0.24816261231899261,
      0.0031020035967230797,
      0.005921422969549894,
      0.004388082772493362,
      0.055432919412851334,
      0.04473433643579483,
      0.004542968701571226,
      0.005998202133923769,
      0.028322862461209297,
      0.0030729302670806646,
      0.005604330450296402,
      0.028071245178580284,
      0.0053604114800691605,
      1.0849775208043866e-05
    ],
    "rationales": [
      "def",
      " clamp",
      "(",
      " lo",
      " hi",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " lo",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " lo",
      "\n",
      " hi"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      5,
      7,
      8,
      9,
      10,
      11,
      12,
      16,
      17,
      18,
      19,
      20,
      25,
      27,
      28,
      35
    ],
    "token": ":"
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.035768963396549225,
      0.4179382622241974,
      0.1153395026922226,
      0.2330932915210724,
      4.083650310349185e-06
    ],
    "rationales": [
      "\n",
      " ",
      " x",
      " >",
      ":"
    ],
    "rationales_indexes": [
      28,
      31,
      33,
      34,
      36
    ],
    "token": "\n"
  },
  "38": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.0050993384793400764,
      0.00397820770740509,
      0.005428003612905741,
      0.4562104046344757,
      0.006895157974213362,
      0.12052132189273834,
      0.02383558824658394,
      0.5205813050270081,
      0.004045045003294945,
      0.002607737435027957,
      0.008380616083741188,
      0.006390439812093973,
      0.006212468724697828,
      3.0137010753605864e-07
    ],
    "rationales": [
      " clamp",
      "x",
      " hi",
      " if",
      " <",
      ":",
      " ",
      " ",
      " ",
      " ",
      " if",
      " x",
      ":",
      "\n"
    ],
    "rationales_indexes": [
      1,
      3,
      7,
      13,
      15,
      17,
      19,
      20,
      25,
      31,
      32,
      33,
      36,
      37
    ],
    "token": " "
  },
  "39": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.9994377493858337
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      38
    ],
    "token": " "
  },
  "4": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.029349440708756447,
      0.312375009059906,
      0.001078846282325685
    ],
    "rationales": [
      "def",
      "(",
      "x"
    ],
    "rationales_indexes": [
      0,
      2,
      3
    ],
    "token": ","
  },
  "40": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.9992586970329285
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      39
    ],
    "token": " "
  },
  "41": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Bool"
    ],
    "probabilities": [
      0.9993951320648193
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      40
    ],
    "token": " "
  },
  "42": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9992117881774902
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      41
    ],
    "token": " "
  },
  "43": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.999313473701477
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      42
    ],
    "token": " "
  },
  "44": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Signature"
    ],
    "probabilities": [
      0.9993201494216919
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      43
    ],
    "token": " "
  },
  "45": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      2.6643416276783682e-05,
      0.009926248341798782,
      0.04954494908452034,
      0.03166690841317177,
      0.017945004627108574,
      0.06782738119363785,
      0.00281427800655365,
      0.005288028623908758,
      3.4294947437452095e-10
    ],
    "rationales": [
      "def",
      " clamp",
      "x",
      " hi",
      "):",
      " lo",
      " return",
      " if",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      3,
      7,
      8,
      16,
      26,
      32,
      44
    ],
    "token": " return"
  },
  "46": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      3.277865835116245e-05,
      0.05681280791759491,
      0.0026613434311002493,
      0.09343945980072021,
      0.11866071820259094,
      0.1699061095714569,
      0.16675271093845367,
      0.18368391692638397,
      0.11110403388738632,
      0.09707991033792496,
      0.20120547711849213,
      0.027586355805397034,
      0.1850951462984085,
      0.18371576070785522,
      1.281581785139707e-10
    ],
    "rationales": [
      "def",
      " clamp",
      " hi",
      "\n",
      " lo",
      ":",
      " ",
      " ",
      " >",
      " hi",
      ":",
      "\n",
      " ",
      " ",
      " return"
    ],
    "rationales_indexes": [
      0,
      1,
      7,
      9,
      16,
      17,
      22,
      23,
      34,
      35,
      36,
      37,
      38,
      44,
      45
    ],
    "token": " hi"
  },
  "47": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.7994871139526367,
      2.1190373445278965e-05
    ],
    "rationales": [
      "\n",
      " hi"
    ],
    "rationales_indexes": [
      37,
      46
    ],
    "token": "\n"
  },
  "48": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.4996776282787323,
      0.22238041460514069,
      0.02885497733950615,
      1.5365132810529758e-07
    ],
    "rationales": [
      " >",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      34,
      43,
      44,
      47
    ],
    "token": " "
  },
  "49": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9995001554489136
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      48
    ],
    "token": " "
  },
  "5": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.00012618517212104052,
      0.00022622391406912357,
      0.0005279182805679739,
      3.654835018096492e-05,
      5.3018502512713894e-06
    ],
    "rationales": [
      "def",
      " clamp",
      "(",
      "x",
      ","
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": " lo"
  },
  "50": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.9993146657943726
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      49
    ],
    "token": " "
  },
  "51": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.8945945501327515,
      2.440485313837115e-10
    ],
    "rationales": [
      " return",
      " "
    ],
    "rationales_indexes": [
      45,
      50
    ],
    "token": " return"
  },
  "52": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.0016305472236126661,
      0.17575038969516754,
      0.10996986925601959,
      0.034860942512750626,
      0.01014358177781105,
      0.0713590756058693,
      1.2895979750737752e-07
    ],
    "rationales": [
      "def",
      "(",
      "x",
      "):",
      " x",
      ":",
      " return"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      8,
      14,
      17,
      51
    ],
    "token": " x"
  },
  "53": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.873725414276123,
      0.420706182718277,
      0.0005932965432293713
    ],
    "rationales": [
      "):",
      "\n",
      " x"
    ],
    "rationales_indexes": [
      8,
      47,
      52
    ],
    "token": "\n"
  },
  "54": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9977667331695557
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      53
    ],
    "token": "\n"
  },
  "6": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.017384113743901253,
      0.15098978579044342,
      3.449321939186234e-09
    ],
    "rationales": [
      "def",
      ",",
      " lo"
    ],
    "rationales_indexes": [
      0,
      4,
      5
    ],
    "token": ","
  },
  "7": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      3.424809983698651e-05,
      0.016634423285722733,
      0.009839892387390137,
      0.007138482760637999,
      0.006075747311115265,
      0.0012520572636276484,
      2.1445332620828594e-08
    ],
    "rationales": [
      "def",
      " clamp",
      "(",
      "x",
      ",",
      " lo",
      ","
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6
    ],
    "token": " hi"
  },
  "8": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.0001514180621597916,
      0.01098689716309309,
      0.006255545653402805,
      0.007073717657476664,
      0.0017207422060891986,
      0.007649451959878206,
      0.00868560653179884,
      9.774125508954512e-09
    ],
    "rationales": [
      "def",
      " clamp",
      "(",
      "x",
      ",",
      " lo",
      ",",
      " hi"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ],
    "token": "):"
  },
  "9": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.1828148066997528,
      0.05232931673526764
    ],
    "rationales": [
      "def",
      "):"
    ],
    "rationales_indexes": [
      0,
      8
    ],
    "token": "\n"
  },
  "_phrase": "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n\n"
}
curl -X POST http://127.0.0.1:5000/prompt -H "Content-Type: application/json"  0.03s user 0.04s system 0% cpu 59.021 total

Sample 2

curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def first_even(nums):\n    for n in nums:\n        if n % 2 == 0:\n            return n\n    return None\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.000260335102211684
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " first"
  },
  "10": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9994294047355652
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      9
    ],
    "token": " "
  },
  "11": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.999504566192627
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      10
    ],
    "token": " "
  },
  "12": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Signature"
    ],
    "probabilities": [
      0.0005232469993643463,
      0.0021847153548151255,
      0.001395359286107123,
      0.0023103051353245974,
      0.0020814030431210995,
      0.001935345120728016,
      0.0003420000139158219,
      0.0011700040195137262,
      0.0008906042203307152,
      0.0007308385684154928,
      0.0014839961659163237,
      1.1450504899812586e-07
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11
    ],
    "token": " for"
  },
  "13": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.021536434069275856,
      0.19508929550647736,
      0.0014583623269572854,
      0.2998480498790741,
      0.018331047147512436,
      0.02427474595606327,
      1.3083024896332063e-05
    ],
    "rationales": [
      "def",
      "(",
      "n",
      "):",
      " ",
      " ",
      " for"
    ],
    "rationales_indexes": [
      0,
      4,
      5,
      7,
      9,
      10,
      12
    ],
    "token": " n"
  },
  "14": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.21552690863609314,
      0.013257203623652458,
      0.008582163602113724,
      0.04410514235496521,
      1.6847065126057714e-05
    ],
    "rationales": [
      "def",
      "ums",
      "\n",
      " for",
      " n"
    ],
    "rationales_indexes": [
      0,
      6,
      8,
      12,
      13
    ],
    "token": " in"
  },
  "15": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.0004339825827628374,
      0.00900706835091114,
      0.01739473082125187,
      0.019116226583719254,
      0.020946884527802467,
      0.030450819060206413,
      0.0022012912668287754,
      0.0036948707420378923,
      0.029700716957449913,
      0.025829119607806206,
      0.025024928152561188,
      0.02614578790962696,
      0.02778536267578602,
      5.0210841436637565e-05,
      3.406596817967511e-07
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " n",
      " in"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14
    ],
    "token": " num"
  },
  "16": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.030277028679847717,
      0.0983782708644867,
      6.540117752917851e-11
    ],
    "rationales": [
      "def",
      "(",
      " num"
    ],
    "rationales_indexes": [
      0,
      4,
      15
    ],
    "token": "s"
  },
  "17": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.011627648957073689,
      0.03791608288884163,
      0.02945183403789997,
      0.03403311222791672,
      0.02193627879023552,
      0.03807234391570091,
      0.5591131448745728,
      0.04835328459739685,
      0.023105556145310402,
      0.11322277784347534,
      0.02231704071164131,
      6.666336048510857e-06
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even",
      "n",
      "ums",
      "):",
      "\n",
      " for",
      " in",
      " num",
      "s"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      5,
      6,
      7,
      8,
      12,
      14,
      15,
      16
    ],
    "token": ":"
  },
  "18": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.3801517188549042,
      0.6082953214645386,
      4.96731445309706e-06
    ],
    "rationales": [
      " n",
      " in",
      ":"
    ],
    "rationales_indexes": [
      13,
      14,
      17
    ],
    "token": "\n"
  },
  "19": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0003379105473868549,
      0.8210347294807434,
      7.247776068197709e-08
    ],
    "rationales": [
      " ",
      ":",
      "\n"
    ],
    "rationales_indexes": [
      11,
      17,
      18
    ],
    "token": " "
  },
  "2": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.0012305463897064328,
      2.2297770740919987e-08
    ],
    "rationales": [
      "def",
      " first"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "_"
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.999420166015625
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      19
    ],
    "token": " "
  },
  "21": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.999428927898407
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      20
    ],
    "token": " "
  },
  "22": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.9994227886199951
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      21
    ],
    "token": " "
  },
  "23": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.999372661113739
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      22
    ],
    "token": " "
  },
  "24": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9994456171989441
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      23
    ],
    "token": " "
  },
  "25": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.9993972778320312
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      24
    ],
    "token": " "
  },
  "26": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.05185310170054436,
      0.12960663437843323,
      0.0005514624645002186,
      0.09484042227268219,
      0.017972588539123535,
      0.09152785688638687,
      0.04783666133880615,
      0.102025106549263,
      0.016407959163188934,
      0.0033262574579566717,
      0.04832153767347336,
      0.05900884047150612,
      0.02882872335612774,
      0.07145511358976364,
      0.014586765319108963,
      1.0718889598138048e-07
    ],
    "rationales": [
      "def",
      " first",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " for",
      " n",
      " in",
      " num",
      "s",
      "\n",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      4,
      5,
      6,
      7,
      8,
      9,
      11,
      12,
      13,
      14,
      15,
      16,
      18,
      25
    ],
    "token": " if"
  },
  "27": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.40580588579177856,
      0.0010088112903758883,
      9.121805487666279e-05
    ],
    "rationales": [
      " n",
      ":",
      " if"
    ],
    "rationales_indexes": [
      13,
      17,
      26
    ],
    "token": " n"
  },
  "28": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.0004226875025779009,
      0.03596998751163483,
      0.004412414971739054,
      0.007789623457938433,
      0.039296284317970276,
      0.04471104219555855,
      0.02100037783384323,
      0.01146724633872509,
      0.006647605448961258,
      0.05135658010840416,
      0.05221223086118698,
      0.05120181664824486,
      0.035378292202949524,
      0.026028823107481003,
      0.04788827896118164,
      0.0023156120441854,
      0.047848887741565704,
      0.03888394683599472,
      0.006089968606829643,
      0.028439441695809364,
      0.02900891751050949,
      0.029573814943432808,
      0.017853770405054092,
      0.029645714908838272,
      0.025582680478692055,
      0.017001209780573845,
      0.006371574010699987,
      9.115890406974358e-07
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " n",
      " in",
      " num",
      "s",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " if",
      " n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27
    ],
    "token": " %"
  },
  "29": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Structural"
    ],
    "probabilities": [
      0.10910486429929733,
      0.05203023552894592,
      0.07699039578437805,
      0.09167548269033432,
      0.08026675134897232,
      0.07732070237398148,
      0.06360123306512833,
      0.02757604792714119,
      0.09373916685581207,
      0.04403064399957657,
      0.07290580123662949,
      0.0833478793501854,
      0.09272009134292603,
      0.2686631679534912,
      0.04324614256620407,
      4.1786435758695006e-05
    ],
    "rationales": [
      "def",
      "(",
      "ums",
      "\n",
      " ",
      " ",
      " ",
      " in",
      " num",
      "s",
      ":",
      "\n",
      " ",
      " if",
      " n",
      " %"
    ],
    "rationales_indexes": [
      0,
      4,
      6,
      8,
      9,
      10,
      11,
      14,
      15,
      16,
      17,
      18,
      19,
      26,
      27,
      28
    ],
    "token": " 2"
  },
  "3": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00010336909326724708,
      2.9085578717058524e-05,
      1.532801263692818e-07
    ],
    "rationales": [
      "def",
      " first",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "even"
  },
  "30": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.00022143036767374724,
      0.012532328255474567,
      0.1033797636628151,
      0.0012931153178215027,
      0.007843323051929474,
      0.0390804298222065,
      1.1111553517573203e-11
    ],
    "rationales": [
      "def",
      "_",
      "n",
      "):",
      " if",
      " %",
      " 2"
    ],
    "rationales_indexes": [
      0,
      2,
      5,
      7,
      26,
      28,
      29
    ],
    "token": " =="
  },
  "31": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.23076209425926208,
      0.03155352175235748,
      0.07946654409170151,
      0.07888446003198624,
      0.07310465723276138,
      0.06268540769815445,
      0.2571704089641571,
      0.052310843020677567,
      6.175902854010928e-06
    ],
    "rationales": [
      " ",
      ":",
      "\n",
      " ",
      " if",
      " n",
      " %",
      " 2",
      " =="
    ],
    "rationales_indexes": [
      10,
      17,
      18,
      19,
      26,
      27,
      28,
      29,
      30
    ],
    "token": " 0"
  },
  "32": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.0031413875985890627,
      0.10217129439115524,
      0.057162363082170486,
      0.11638185381889343,
      0.3269031047821045,
      0.00022071025159675628
    ],
    "rationales": [
      "def",
      "even",
      "):",
      " for",
      " n",
      " 0"
    ],
    "rationales_indexes": [
      0,
      3,
      7,
      12,
      13,
      31
    ],
    "token": ":"
  },
  "33": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.2314353883266449,
      0.5181984305381775,
      0.3010779917240143,
      2.7096766643808223e-06
    ],
    "rationales": [
      " n",
      " in",
      ":",
      ":"
    ],
    "rationales_indexes": [
      13,
      14,
      17,
      32
    ],
    "token": "\n"
  },
  "34": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.06559005379676819,
      0.2561754286289215,
      0.0007493480807170272,
      5.075035574009235e-07
    ],
    "rationales": [
      " num",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      15,
      24,
      25,
      33
    ],
    "token": " "
  },
  "35": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9993482232093811
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      34
    ],
    "token": " "
  },
  "36": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9992576241493225
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      35
    ],
    "token": " "
  },
  "37": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Asserts"
    ],
    "probabilities": [
      0.9992762207984924
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      36
    ],
    "token": " "
  },
  "38": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.9992896318435669
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      37
    ],
    "token": " "
  },
  "39": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9994377493858337
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      38
    ],
    "token": " "
  },
  "4": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.008550094440579414,
      0.005527399014681578,
      0.010531415231525898,
      0.0006299103260971606
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "("
  },
  "40": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9992586970329285
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      39
    ],
    "token": " "
  },
  "41": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.9993951320648193
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      40
    ],
    "token": " "
  },
  "42": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.9992117881774902
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      41
    ],
    "token": " "
  },
  "43": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.999313473701477
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      42
    ],
    "token": " "
  },
  "44": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9993201494216919
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      43
    ],
    "token": " "
  },
  "45": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      2.6643416276783682e-05,
      0.10032309591770172,
      0.001264816033653915,
      0.02620021253824234,
      0.01568538136780262,
      0.004662888124585152,
      0.0002437201328575611,
      0.039568766951560974,
      0.0644800066947937,
      3.4294947437452095e-10
    ],
    "rationales": [
      "def",
      "_",
      "):",
      "\n",
      " ",
      " for",
      " if",
      " ==",
      ":",
      " "
    ],
    "rationales_indexes": [
      0,
      2,
      7,
      8,
      11,
      12,
      26,
      30,
      32,
      44
    ],
    "token": " return"
  },
  "46": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      0.0007556337513960898,
      0.018661588430404663,
      0.05786773934960365,
      0.007085006218403578,
      0.11572915315628052,
      8.485661595614147e-08
    ],
    "rationales": [
      "def",
      " first",
      "even",
      " n",
      ":",
      " return"
    ],
    "rationales_indexes": [
      0,
      1,
      3,
      13,
      17,
      45
    ],
    "token": " n"
  },
  "47": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.21946589648723602,
      0.37057849764823914,
      0.4331819415092468,
      0.004641635809093714
    ],
    "rationales": [
      "_",
      " n",
      "\n",
      " n"
    ],
    "rationales_indexes": [
      2,
      27,
      33,
      46
    ],
    "token": "\n"
  },
  "48": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Types"
    ],
    "probabilities": [
      0.6446377635002136,
      0.22238041460514069,
      0.02885497733950615,
      1.5365132810529758e-07
    ],
    "rationales": [
      " num",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      15,
      43,
      44,
      47
    ],
    "token": " "
  },
  "49": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9995001554489136
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      48
    ],
    "token": " "
  },
  "5": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.009041239507496357,
      0.016304664313793182,
      0.013779032044112682,
      0.011635822243988514,
      4.875322701991536e-05
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "n"
  },
  "50": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.9993146657943726
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      49
    ],
    "token": " "
  },
  "51": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.8945945501327515,
      2.440485313837115e-10
    ],
    "rationales": [
      " return",
      " "
    ],
    "rationales_indexes": [
      45,
      50
    ],
    "token": " return"
  },
  "52": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.005786196794360876,
      0.008838651701807976,
      0.000632945797406137,
      0.014977442100644112,
      0.008174404501914978,
      0.013660947792232037,
      0.017870651558041573,
      0.00838124006986618,
      0.006685164757072926,
      0.01406500767916441,
      0.01185520738363266,
      0.012806132435798645,
      0.010576602071523666,
      0.005697397515177727,
      0.006949894595891237,
      0.014004114083945751,
      0.007295256946235895,
      0.009169811382889748,
      0.01504222396761179,
      0.021940244361758232,
      0.014246756210923195,
      0.010292330756783485,
      0.009046548046171665,
      0.00960170105099678,
      0.008346459828317165,
      0.00663954671472311,
      0.006409383378922939,
      0.007524767424911261,
      0.005732019431889057,
      0.013152983970940113,
      0.014712396077811718,
      0.009275490418076515,
      0.006789295934140682,
      0.016143424436450005,
      0.025114232674241066,
      0.02450326643884182,
      0.009149078279733658,
      0.01929856650531292,
      0.015623072162270546,
      0.02073054015636444,
      0.016892941668629646,
      0.02318260818719864,
      0.0246526338160038,
      0.02525368332862854,
      0.022998619824647903,
      0.008648434653878212,
      0.012891204096376896,
      0.004774966277182102,
      0.020011629909276962,
      0.007906869985163212,
      0.015417643822729588,
      1.7659154138982558e-07
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " n",
      " in",
      " num",
      "s",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " if",
      " n",
      " %",
      " 2",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " n",
      "\n",
      " ",
      " ",
      " ",
      " return"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51
    ],
    "token": " None"
  },
  "53": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.5852596759796143,
      0.0004869343829341233
    ],
    "rationales": [
      "\n",
      " None"
    ],
    "rationales_indexes": [
      47,
      52
    ],
    "token": "\n"
  },
  "54": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9977667331695557
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      53
    ],
    "token": "\n"
  },
  "6": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0001163195411209017,
      0.0031785862520337105,
      0.004794422071427107,
      0.005404170602560043,
      0.0005839170189574361,
      4.1550272555923584e-08
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even",
      "(",
      "n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "token": "ums"
  },
  "7": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0009640572825446725,
      0.011642551980912685,
      0.014515725895762444,
      0.01832539029419422,
      0.0039014145731925964,
      0.0221039317548275,
      3.4326892439651147e-09
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even",
      "(",
      "n",
      "ums"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6
    ],
    "token": "):"
  },
  "8": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.1742928922176361,
      0.06658641248941422
    ],
    "rationales": [
      "def",
      "):"
    ],
    "rationales_indexes": [
      0,
      7
    ],
    "token": "\n"
  },
  "9": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      5.984272206660535e-07,
      7.76812794356374e-06,
      2.8749420835083583e-06,
      9.31134582060622e-06,
      5.234428681433201e-05,
      7.861364792915992e-06,
      1.6059404515544884e-05,
      2.547769554439583e-06,
      1.7788709172350536e-08
    ],
    "rationales": [
      "def",
      " first",
      "_",
      "even",
      "(",
      "n",
      "ums",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    "token": " "
  },
  "_phrase": "def first_even(nums):\n    for n in nums:\n        if n % 2 == 0:\n            return n\n    return None\n\n"
}


Sample 3

curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def safe_int(s):\n    try:\n        return int(s)\n    except ValueError:\n        return None\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      3.618669506977312e-05
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " safe"
  },
  "10": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.9994294047355652
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      9
    ],
    "token": " "
  },
  "11": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      2.367065280850511e-05,
      0.0002436565118841827,
      0.00016384609625674784,
      0.00016943097580224276,
      9.521674655843526e-05,
      0.000211515580303967,
      5.71909113205038e-05,
      9.326938743470237e-05,
      0.00012245995458215475,
      2.7735903131542727e-05,
      8.624219116626364e-10
    ],
    "rationales": [
      "def",
      " safe",
      "_",
      "int",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10
    ],
    "token": " try"
  },
  "12": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.022078627720475197,
      0.4202505946159363,
      3.3956850529648364e-05
    ],
    "rationales": [
      "def",
      "):",
      " try"
    ],
    "rationales_indexes": [
      0,
      6,
      11
    ],
    "token": ":"
  },
  "13": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.21217456459999084,
      0.5673020482063293,
      5.03394676343305e-06
    ],
    "rationales": [
      "):",
      "\n",
      ":"
    ],
    "rationales_indexes": [
      6,
      7,
      12
    ],
    "token": "\n"
  },
  "14": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.004847760312259197,
      0.9355676174163818,
      2.6351035131710887e-08
    ],
    "rationales": [
      " ",
      ":",
      "\n"
    ],
    "rationales_indexes": [
      10,
      12,
      13
    ],
    "token": " "
  },
  "15": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.9994822144508362
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      14
    ],
    "token": " "
  },
  "16": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Functional"
    ],
    "probabilities": [
      0.9994900226593018
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      15
    ],
    "token": " "
  },
  "17": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9994862079620361
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      16
    ],
    "token": " "
  },
  "18": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9994465708732605
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      17
    ],
    "token": " "
  },
  "19": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.9994450211524963
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      18
    ],
    "token": " "
  },
  "2": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.11253375560045242,
      6.829256715690235e-09
    ],
    "rationales": [
      "def",
      " safe"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "_"
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.999420166015625
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      19
    ],
    "token": " "
  },
  "21": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      3.752419070224278e-05,
      0.03977956995368004,
      0.007362696807831526,
      0.0006780879921279848,
      0.12227746844291687,
      5.077991360735723e-10
    ],
    "rationales": [
      "def",
      "):",
      " ",
      " try",
      ":",
      " "
    ],
    "rationales_indexes": [
      0,
      6,
      10,
      11,
      12,
      20
    ],
    "token": " return"
  },
  "22": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.0008723456412553787,
      0.007208202034235001,
      0.009015269577503204,
      0.0025489013642072678,
      0.060214053839445114,
      0.045262254774570465,
      0.02242480218410492,
      0.08961155265569687,
      0.09690705686807632,
      0.09116894006729126,
      0.09018907696008682,
      0.026265738531947136,
      0.08685485273599625,
      0.08136259764432907,
      0.07577203959226608,
      0.06205279007554054,
      0.05016214773058891,
      0.03639906644821167,
      0.03246593475341797,
      0.04185978323221207,
      0.06451002508401871,
      1.9892627278750297e-07
    ],
    "rationales": [
      "def",
      " safe",
      "_",
      "int",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " try",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21
    ],
    "token": " int"
  },
  "23": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0033147423528134823,
      0.08484454452991486,
      0.008001051843166351,
      0.05167410895228386,
      0.01539948396384716,
      0.1824587732553482,
      1.3674711851763277e-07
    ],
    "rationales": [
      "def",
      "_",
      "int",
      "):",
      " try",
      " return",
      " int"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      6,
      11,
      21,
      22
    ],
    "token": "("
  },
  "24": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.0026569771580398083,
      0.10929463058710098,
      0.0052080885507166386,
      0.0019095356110483408,
      7.207597082015127e-05
    ],
    "rationales": [
      "def",
      "(",
      "s",
      ":",
      "("
    ],
    "rationales_indexes": [
      0,
      4,
      5,
      12,
      23
    ],
    "token": "s"
  },
  "25": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.018352994695305824,
      0.1916239857673645,
      0.12079015374183655,
      9.756283361639362e-06
    ],
    "rationales": [
      "def",
      "int",
      "(",
      "s"
    ],
    "rationales_indexes": [
      0,
      3,
      23,
      24
    ],
    "token": ")"
  },
  "26": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.8683161735534668,
      0.07209721207618713
    ],
    "rationales": [
      "s",
      ")"
    ],
    "rationales_indexes": [
      24,
      25
    ],
    "token": "\n"
  },
  "27": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Structural"
    ],
    "probabilities": [
      0.811147153377533,
      0.025222299620509148,
      0.0006889365031383932,
      0.27287358045578003,
      2.5658553681751073e-07
    ],
    "rationales": [
      " ",
      " ",
      " ",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      18,
      19,
      20,
      21,
      26
    ],
    "token": " "
  },
  "28": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.9993835687637329
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      27
    ],
    "token": " "
  },
  "29": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9992671608924866
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      28
    ],
    "token": " "
  },
  "3": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.0025702177081257105,
      0.0015646099345758557,
      5.1509694287688035e-08
    ],
    "rationales": [
      "def",
      " safe",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "int"
  },
  "30": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      5.271480040391907e-05,
      0.01972934789955616,
      0.001429603318683803,
      0.00011692300904542208,
      3.7914494441793067e-06,
      0.00024683497031219304,
      0.00019293850346002728,
      0.00035442705848254263,
      0.0025205386336892843,
      0.0025990507565438747,
      0.001993420533835888,
      0.0001780746824806556,
      0.0019761514849960804,
      0.00024123112962115556,
      0.000269094918621704,
      0.0021536487620323896,
      9.788315219338983e-05,
      0.00019851801334880292,
      0.00036318617640063167,
      0.00029318087035790086,
      0.0002459184906911105,
      0.001392658450640738,
      0.0021453755907714367,
      0.025551505386829376,
      0.0008572131046094,
      9.073230467038229e-05,
      4.600730972015299e-05,
      0.00743411760777235,
      0.0003630883584264666,
      1.8954103564450264e-10
    ],
    "rationales": [
      "def",
      " safe",
      "_",
      "int",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " try",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " int",
      "(",
      "s",
      ")",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29
    ],
    "token": " except"
  },
  "31": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.0001387814845656976,
      0.018116338178515434,
      0.11130522191524506,
      0.041507504880428314,
      0.026923134922981262,
      1.9367908166145753e-08
    ],
    "rationales": [
      "def",
      "_",
      "int",
      "(",
      " try",
      " except"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      4,
      11,
      30
    ],
    "token": " Value"
  },
  "32": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      2.2097181499702856e-05,
      0.3773624897003174,
      0.2029743492603302,
      0.004651633556932211,
      0.03275574371218681,
      2.8059967543958564e-12
    ],
    "rationales": [
      "def",
      "_",
      "):",
      "(",
      " except",
      " Value"
    ],
    "rationales_indexes": [
      0,
      2,
      6,
      23,
      30,
      31
    ],
    "token": "Error"
  },
  "33": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.3008546233177185,
      5.129068995302077e-06
    ],
    "rationales": [
      "def",
      "Error"
    ],
    "rationales_indexes": [
      0,
      32
    ],
    "token": ":"
  },
  "34": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Bool"
    ],
    "probabilities": [
      0.23587994277477264,
      0.05030493810772896,
      0.5334333777427673,
      4.131927653361345e-06
    ],
    "rationales": [
      " return",
      ")",
      "\n",
      ":"
    ],
    "rationales_indexes": [
      21,
      25,
      26,
      33
    ],
    "token": "\n"
  },
  "35": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.4847092926502228,
      0.2716844975948334,
      0.004378346726298332,
      0.5433889627456665,
      0.09191230684518814,
      3.9820162101023016e-07
    ],
    "rationales": [
      " ",
      " ",
      " ",
      " Value",
      ":",
      "\n"
    ],
    "rationales_indexes": [
      27,
      28,
      29,
      31,
      33,
      34
    ],
    "token": " "
  },
  "36": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.9992576241493225
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      35
    ],
    "token": " "
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.9992762207984924
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      36
    ],
    "token": " "
  },
  "38": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.9992896318435669
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      37
    ],
    "token": " "
  },
  "39": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9994377493858337
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      38
    ],
    "token": " "
  },
  "4": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.11719443649053574,
      1.3043581326144249e-08
    ],
    "rationales": [
      "def",
      "int"
    ],
    "rationales_indexes": [
      0,
      3
    ],
    "token": "("
  },
  "40": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "OOP"
    ],
    "probabilities": [
      0.9992586970329285
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      39
    ],
    "token": " "
  },
  "41": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "With"
    ],
    "probabilities": [
      0.9993951320648193
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      40
    ],
    "token": " "
  },
  "42": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      2.4557200958952308e-05,
      0.002090800553560257,
      0.01240451354533434,
      0.0644497498869896,
      0.003340406809002161,
      0.022966155782341957,
      0.16496430337429047,
      5.722226581461598e-10
    ],
    "rationales": [
      "def",
      " return",
      ")",
      " ",
      " except",
      "Error",
      ":",
      " "
    ],
    "rationales_indexes": [
      0,
      21,
      25,
      29,
      30,
      32,
      33,
      41
    ],
    "token": " return"
  },
  "43": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.06001671776175499,
      0.0006463233730755746,
      0.00881337933242321,
      0.03766312450170517,
      0.19442522525787354,
      0.015870433300733566,
      0.05563125014305115,
      0.05529200658202171,
      0.014153947122395039,
      0.06179802492260933,
      0.06259464472532272,
      0.022583598271012306,
      0.03242915868759155,
      0.05176866054534912,
      0.014996117912232876,
      0.0017689340747892857,
      0.02438383921980858,
      1.533453968249887e-07
    ],
    "rationales": [
      "def",
      "_",
      "(",
      "s",
      "):",
      " ",
      " ",
      " ",
      " try",
      ":",
      " ",
      " int",
      "s",
      " except",
      " Value",
      "Error",
      ":",
      " return"
    ],
    "rationales_indexes": [
      0,
      2,
      4,
      5,
      6,
      8,
      9,
      10,
      11,
      12,
      15,
      22,
      24,
      30,
      31,
      32,
      33,
      42
    ],
    "token": " None"
  },
  "44": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      0.6406023502349854,
      0.0005711985286325216
    ],
    "rationales": [
      "\n",
      " None"
    ],
    "rationales_indexes": [
      34,
      43
    ],
    "token": "\n"
  },
  "45": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.9962778687477112
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      44
    ],
    "token": "\n"
  },
  "46": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.0006416878895834088,
      0.09494584053754807,
      1.0473745204464535e-09
    ],
    "rationales": [
      "def",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      0,
      42,
      45
    ],
    "token": "return"
  },
  "47": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.026003265753388405,
      0.003837556578218937,
      0.002371430629864335,
      0.024308137595653534,
      0.0031056483276188374,
      0.03918115422129631,
      0.019484292715787888,
      0.047335319221019745,
      0.03855825588107109,
      0.04423307627439499,
      0.012936512939631939,
      0.007878447882831097,
      0.0833611711859703,
      0.030625086277723312,
      0.019639864563941956,
      0.0009759743697941303,
      0.0007546440465375781,
      0.03343518078327179,
      0.045724667608737946,
      0.046201907098293304,
      0.036194320768117905,
      0.0324917696416378,
      0.00041497524944134057,
      0.0035445375833660364,
      0.0045357332564890385,
      0.012834648601710796,
      0.013311587274074554,
      6.357038984106111e-08
    ],
    "rationales": [
      " ",
      " ",
      " ",
      " try",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " int",
      "s",
      "\n",
      " ",
      "Error",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " None",
      "\n",
      "return"
    ],
    "rationales_indexes": [
      8,
      9,
      10,
      11,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      24,
      26,
      27,
      32,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      45,
      46
    ],
    "token": " int"
  },
  "48": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.002915977966040373,
      0.09074495732784271,
      0.037267424166202545,
      0.06103663891553879,
      0.03177454322576523,
      0.07086019963026047,
      0.01891341619193554,
      9.786206334183589e-08
    ],
    "rationales": [
      "def",
      "int",
      "(",
      "s",
      "):",
      " try",
      ")",
      " int"
    ],
    "rationales_indexes": [
      0,
      3,
      4,
      5,
      6,
      11,
      25,
      47
    ],
    "token": "("
  },
  "49": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.15514050424098969,
      0.01571069471538067,
      0.00984877161681652,
      0.008019506931304932,
      0.004361503757536411,
      0.39797666668891907,
      0.03485608845949173,
      0.04560001194477081,
      0.02070854790508747,
      6.966658111196011e-05
    ],
    "rationales": [
      " return",
      "(",
      "s",
      ")",
      ":",
      "\n",
      "\n",
      "\n",
      " int",
      "("
    ],
    "rationales_indexes": [
      21,
      23,
      24,
      25,
      33,
      34,
      44,
      45,
      47,
      48
    ],
    "token": "s"
  },
  "5": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.003419723128899932,
      0.009196115657687187,
      0.010522016324102879,
      0.008585070259869099,
      8.780310599831864e-05
    ],
    "rationales": [
      "def",
      " safe",
      "_",
      "int",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "s"
  },
  "50": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Loops"
    ],
    "probabilities": [
      0.00900288950651884,
      0.1779586225748062,
      0.028513485565781593,
      0.12586888670921326,
      5.6036700698314235e-05
    ],
    "rationales": [
      "def",
      "s",
      "):",
      "(",
      "s"
    ],
    "rationales_indexes": [
      0,
      5,
      6,
      48,
      49
    ],
    "token": ")"
  },
  "51": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "With"
    ],
    "probabilities": [
      0.6677719950675964,
      0.0365624874830246
    ],
    "rationales": [
      " return",
      ")"
    ],
    "rationales_indexes": [
      21,
      50
    ],
    "token": "\n"
  },
  "52": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.9971799850463867
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      51
    ],
    "token": "\n"
  },
  "53": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.0005750640411861241,
      0.06470806896686554,
      0.009628953412175179,
      0.15738974511623383,
      1.2052621123359586e-09
    ],
    "rationales": [
      "def",
      "_",
      "):",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      0,
      2,
      6,
      42,
      52
    ],
    "token": "def"
  },
  "54": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.001012704218737781,
      0.00953607726842165,
      0.09318318217992783,
      0.20873907208442688,
      0.24763831496238708,
      0.004504027776420116,
      0.03578963130712509,
      0.0613352432847023,
      0.08282621949911118,
      0.12959963083267212,
      0.23740503191947937,
      0.27529376745224,
      0.27295851707458496,
      0.2627940773963928,
      0.27643710374832153,
      0.2850476801395416,
      0.15206359326839447,
      0.01951095461845398,
      0.0732022300362587,
      0.11043210327625275,
      0.17736537754535675,
      0.046476516872644424,
      4.35366018791683e-08
    ],
    "rationales": [
      "def",
      "_",
      "int",
      "(",
      "s",
      "):",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " int",
      " ",
      " ",
      " ",
      " except",
      "\n",
      " ",
      " ",
      " None",
      "def"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      4,
      5,
      6,
      8,
      9,
      10,
      14,
      15,
      16,
      17,
      22,
      27,
      28,
      29,
      30,
      34,
      40,
      41,
      43,
      53
    ],
    "token": " __"
  },
  "6": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.0023190281353890896,
      0.03175082802772522,
      0.0276899766176939,
      0.01939569041132927,
      0.014393526129424572,
      2.7063149943273856e-08
    ],
    "rationales": [
      "def",
      " safe",
      "_",
      "int",
      "(",
      "s"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "token": "):"
  },
  "7": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Types"
    ],
    "probabilities": [
      0.8355810642242432,
      0.06029180809855461
    ],
    "rationales": [
      "s",
      "):"
    ],
    "rationales_indexes": [
      5,
      6
    ],
    "token": "\n"
  },
  "8": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      1.6640514104437898e-06,
      2.1097957869642414e-06,
      2.2136202915135073e-06,
      2.443267931084847e-06,
      4.3014697439502925e-05,
      2.9778724638163112e-05,
      2.3235572371049784e-05,
      1.7774368643586058e-08
    ],
    "rationales": [
      "def",
      " safe",
      "_",
      "int",
      "(",
      "s",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ],
    "token": " "
  },
  "9": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.999458372592926
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      8
    ],
    "token": " "
  },
  "_phrase": "def safe_int(s):\n    try:\n        return int(s)\n    except ValueError:\n        return None\n\nreturn int(s)\n\ndef __"
}



Sample 4

time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def add_count(d, key):\n    d[key] = d.get(key, 0) + 1\n    return d[key]\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.00010902214125962928
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " add"
  },
  "10": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      2.3815257463866146e-06,
      1.3494091945176478e-05,
      8.016233186936006e-05,
      3.80320125259459e-05,
      6.36122131254524e-05,
      2.6496600185055286e-05,
      5.405450065154582e-05,
      4.6420685976045206e-05,
      8.357176557183266e-05,
      1.4153983762810185e-08
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count",
      "(",
      "d",
      ",",
      " key",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9
    ],
    "token": " "
  },
  "11": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.999504566192627
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      10
    ],
    "token": " "
  },
  "12": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.9994699358940125
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      11
    ],
    "token": " "
  },
  "13": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.00012366828741505742,
      0.00034307350870221853,
      0.0006604768568649888,
      0.0010444067884236574,
      0.0006134612485766411,
      0.0005737251485697925,
      0.001090365112759173,
      0.0013868657406419516,
      0.0009365010773763061,
      0.0013274288503453135,
      0.00022794876713305712,
      0.0008267532102763653,
      7.214892434603826e-07
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count",
      "(",
      "d",
      ",",
      " key",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12
    ],
    "token": " d"
  },
  "14": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.002113100839778781,
      0.11840616166591644,
      0.11069439351558685,
      0.045607056468725204,
      0.09414642304182053,
      0.04545460641384125,
      0.08412859588861465,
      0.11154025048017502,
      0.03552985563874245,
      0.03576549515128136,
      0.03500483185052872,
      0.03402937203645706,
      0.03413045406341553,
      6.073755387525637e-10
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count",
      "(",
      "d",
      ",",
      " key",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " d"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13
    ],
    "token": "["
  },
  "15": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0001545295090181753,
      0.07544680684804916,
      0.020361974835395813,
      0.0741720050573349,
      0.048968978226184845,
      0.05133206024765968,
      0.005805102176964283,
      0.06683045625686646,
      1.7007628230203409e-06
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count",
      "d",
      ",",
      " key",
      "):",
      "["
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      5,
      6,
      7,
      8,
      14
    ],
    "token": "key"
  },
  "16": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.2957290709018707,
      0.11033599823713303,
      5.2601051986345126e-11
    ],
    "rationales": [
      "def",
      "[",
      "key"
    ],
    "rationales_indexes": [
      0,
      14,
      15
    ],
    "token": "]"
  },
  "17": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.015552343800663948,
      0.1155167892575264,
      0.0608983039855957,
      0.3741186857223511,
      0.11944679170846939,
      0.00032902671955525875
    ],
    "rationales": [
      "def",
      "_",
      "count",
      ",",
      " key",
      "]"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      6,
      7,
      16
    ],
    "token": " ="
  },
  "18": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.0404917411506176,
      0.9917801022529602,
      0.00012917233107145876
    ],
    "rationales": [
      " d",
      "key",
      " ="
    ],
    "rationales_indexes": [
      13,
      15,
      17
    ],
    "token": " d"
  },
  "19": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Structural"
    ],
    "probabilities": [
      0.013339255936443806,
      0.06413110345602036,
      0.027676193043589592,
      0.16880057752132416,
      4.1676051409922366e-07
    ],
    "rationales": [
      "def",
      "_",
      "):",
      " =",
      " d"
    ],
    "rationales_indexes": [
      0,
      2,
      8,
      17,
      18
    ],
    "token": "."
  },
  "2": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.16413041949272156,
      2.6517929518909966e-10
    ],
    "rationales": [
      "def",
      " add"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "_"
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.00024136692809406668,
      0.015401934273540974,
      0.005687421187758446,
      0.03202918916940689,
      0.10363888740539551,
      0.02456432580947876,
      0.08690967410802841,
      0.035673972219228745,
      1.109350655781327e-08
    ],
    "rationales": [
      "def",
      " add",
      "count",
      ",",
      "[",
      "]",
      " =",
      " d",
      "."
    ],
    "rationales_indexes": [
      0,
      1,
      3,
      6,
      14,
      16,
      17,
      18,
      19
    ],
    "token": "get"
  },
  "21": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.016501111909747124,
      0.05899251252412796,
      0.2208530157804489,
      0.00017991868662647903
    ],
    "rationales": [
      "def",
      "):",
      "key",
      "get"
    ],
    "rationales_indexes": [
      0,
      8,
      15,
      20
    ],
    "token": "("
  },
  "22": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0003466439666226506,
      0.017105353996157646,
      0.05645129457116127,
      3.959862127089764e-08
    ],
    "rationales": [
      "def",
      " key",
      "):",
      "("
    ],
    "rationales_indexes": [
      0,
      7,
      8,
      21
    ],
    "token": "key"
  },
  "23": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.06464724242687225,
      0.16079598665237427,
      0.21057304739952087,
      0.022417407482862473,
      2.2390834075736166e-08
    ],
    "rationales": [
      "(",
      ",",
      "[",
      ".",
      "key"
    ],
    "rationales_indexes": [
      4,
      6,
      14,
      19,
      22
    ],
    "token": ","
  },
  "24": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.009266311302781105,
      0.00732372235506773,
      0.025286249816417694,
      0.003951722290366888,
      0.003317883238196373,
      0.09472007304430008,
      0.0017902209656313062,
      0.016088157892227173,
      0.06131603568792343,
      0.07879842072725296,
      0.061738528311252594,
      0.03097878210246563,
      0.030062654986977577,
      1.867605169536546e-05
    ],
    "rationales": [
      "def",
      "_",
      "count",
      " key",
      " d",
      "[",
      "key",
      " =",
      " d",
      ".",
      "get",
      "(",
      "key",
      ","
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      7,
      13,
      14,
      15,
      17,
      18,
      19,
      20,
      21,
      22,
      23
    ],
    "token": " 0"
  },
  "25": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.010658448562026024,
      0.2664312422275543,
      0.2226630598306656,
      1.5073652321007103e-05
    ],
    "rationales": [
      "def",
      "count",
      "(",
      " 0"
    ],
    "rationales_indexes": [
      0,
      3,
      4,
      24
    ],
    "token": ")"
  },
  "26": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Exceptions"
    ],
    "probabilities": [
      0.008135119453072548,
      0.052174847573041916,
      0.03377241641283035,
      0.014501367695629597,
      0.0636662095785141,
      3.0555675039067864e-05
    ],
    "rationales": [
      "def",
      "count",
      "(",
      "key",
      " 0",
      ")"
    ],
    "rationales_indexes": [
      0,
      3,
      21,
      22,
      24,
      25
    ],
    "token": " +"
  },
  "27": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.1371709406375885,
      0.0933322012424469,
      0.11657651513814926,
      0.05350848659873009,
      0.1247178390622139,
      0.08158320933580399,
      0.138737291097641,
      0.06467919796705246,
      0.0009717054781503975
    ],
    "rationales": [
      "d",
      " ",
      " d",
      "]",
      " =",
      " d",
      "get",
      ")",
      " +"
    ],
    "rationales_indexes": [
      5,
      10,
      13,
      16,
      17,
      18,
      20,
      25,
      26
    ],
    "token": " 1"
  },
  "28": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.05175651237368584,
      0.26230645179748535,
      0.25111261010169983,
      0.23057198524475098,
      0.00016154984768945724
    ],
    "rationales": [
      "def",
      "count",
      " key",
      "\n",
      " 1"
    ],
    "rationales_indexes": [
      0,
      3,
      7,
      9,
      27
    ],
    "token": "\n"
  },
  "29": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.005446425639092922,
      0.004718754440546036,
      0.01023910753428936,
      0.22997592389583588,
      0.0004465373058337718,
      0.026405420154333115,
      0.0029900209046900272,
      0.012642293237149715,
      0.005527161993086338,
      0.011748277582228184,
      0.001354252570308745,
      0.002599085448309779,
      0.0027197585441172123,
      0.005972005892544985,
      0.003137057414278388,
      2.8258369866307476e-07
    ],
    "rationales": [
      "d",
      "):",
      " ",
      " ",
      " ",
      "[",
      "]",
      " d",
      ".",
      "get",
      "key",
      ",",
      " 0",
      " +",
      " 1",
      "\n"
    ],
    "rationales_indexes": [
      5,
      8,
      10,
      11,
      12,
      14,
      16,
      18,
      19,
      20,
      22,
      23,
      24,
      26,
      27,
      28
    ],
    "token": " "
  },
  "3": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.004341209307312965,
      0.001883185002952814,
      1.2313302022448624e-07
    ],
    "rationales": [
      "def",
      " add",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "count"
  },
  "30": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9993077516555786
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      29
    ],
    "token": " "
  },
  "31": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9993191957473755
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      30
    ],
    "token": " "
  },
  "32": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0002832065802067518,
      0.13828857243061066,
      0.11764528602361679,
      0.1335485726594925,
      0.1059783548116684,
      0.12025704979896545,
      0.0029244734905660152,
      0.004895108751952648,
      0.015445093624293804,
      0.12472100555896759,
      0.0007421906339004636,
      0.0011402728268876672,
      0.0005215818528085947,
      0.0012817519018426538,
      4.3610602006083354e-05,
      0.0003960627946071327,
      0.07679431140422821,
      0.00013176974607631564,
      0.0008825141121633351,
      0.0004488632548600435,
      0.00043651313171721995,
      0.000773076550103724,
      0.0008877115906216204,
      0.0007351007079705596,
      0.001124379807151854,
      0.045565005391836166,
      0.0002996059483848512,
      0.0008830265142023563,
      0.0010668239556252956,
      0.0005836375639773905,
      0.00034007334033958614,
      7.010132474505326e-10
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count",
      "(",
      "d",
      ",",
      " key",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " d",
      "[",
      "key",
      "]",
      " =",
      " d",
      ".",
      "get",
      "(",
      "key",
      ",",
      " 0",
      ")",
      " +",
      " 1",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31
    ],
    "token": " return"
  },
  "33": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.0021447306498885155,
      0.09439032524824142,
      0.02600705809891224,
      1.6742460218210908e-07
    ],
    "rationales": [
      "def",
      "count",
      " d",
      " return"
    ],
    "rationales_indexes": [
      0,
      3,
      13,
      32
    ],
    "token": " d"
  },
  "34": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.019307302311062813,
      0.012751988135278225,
      0.31055593490600586,
      0.001936861895956099,
      0.013734587468206882,
      0.06756892055273056,
      0.498992919921875,
      0.037832796573638916,
      4.753186177985924e-10
    ],
    "rationales": [
      "def",
      "\n",
      " d",
      "[",
      "]",
      ".",
      "(",
      "\n",
      " d"
    ],
    "rationales_indexes": [
      0,
      9,
      13,
      14,
      16,
      19,
      21,
      28,
      33
    ],
    "token": "["
  },
  "35": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      4.2522231524344534e-05,
      0.05645773559808731,
      0.001978886080905795,
      0.013641959987580776,
      0.11106082797050476,
      1.1677772135954e-06
    ],
    "rationales": [
      "def",
      " add",
      " key",
      "]",
      " return",
      "["
    ],
    "rationales_indexes": [
      0,
      1,
      7,
      16,
      32,
      34
    ],
    "token": "key"
  },
  "36": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.3261739909648895,
      9.211337942405251e-11
    ],
    "rationales": [
      "[",
      "key"
    ],
    "rationales_indexes": [
      34,
      35
    ],
    "token": "]"
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.32708460092544556,
      0.002038884675130248
    ],
    "rationales": [
      "def",
      "]"
    ],
    "rationales_indexes": [
      0,
      36
    ],
    "token": "\n"
  },
  "38": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.9953950047492981
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      37
    ],
    "token": "\n"
  },
  "39": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.0008602691814303398,
      0.178233340382576,
      0.07239289581775665,
      0.016342701390385628,
      2.4347301952332145e-09
    ],
    "rationales": [
      "def",
      "_",
      "):",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      0,
      2,
      8,
      32,
      38
    ],
    "token": "def"
  },
  "4": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Types"
    ],
    "probabilities": [
      0.03856496140360832,
      0.11683041602373123,
      0.11291944980621338,
      1.26923634979903e-06
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "("
  },
  "40": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.003007233841344714,
      0.2647998332977295,
      0.001657392829656601,
      0.0013089175336062908,
      0.000648894754704088,
      0.001796918106265366,
      0.0016316291876137257,
      0.00013635162031278014,
      0.0006116755539551377,
      0.0006686587585136294,
      3.961269534613621e-09
    ],
    "rationales": [
      "def",
      " add",
      " d",
      "]",
      " d",
      ".",
      "key",
      "]",
      "\n",
      "\n",
      "def"
    ],
    "rationales_indexes": [
      0,
      1,
      13,
      16,
      18,
      19,
      22,
      36,
      37,
      38,
      39
    ],
    "token": " add"
  },
  "41": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.031440768390893936,
      0.14854981005191803,
      7.211314656530021e-09
    ],
    "rationales": [
      "def",
      "_",
      " add"
    ],
    "rationales_indexes": [
      0,
      2,
      40
    ],
    "token": "_"
  },
  "42": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.23647677898406982,
      0.7082080841064453,
      8.932057227184487e-08
    ],
    "rationales": [
      "count",
      "d",
      "_"
    ],
    "rationales_indexes": [
      3,
      5,
      41
    ],
    "token": "count"
  },
  "43": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.12853695452213287,
      0.026277577504515648,
      3.4726699027487484e-07
    ],
    "rationales": [
      "def",
      ")",
      "count"
    ],
    "rationales_indexes": [
      0,
      25,
      42
    ],
    "token": "("
  },
  "44": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Structural"
    ],
    "probabilities": [
      0.07073313742876053,
      0.24201589822769165,
      0.00016116886399686337
    ],
    "rationales": [
      " d",
      "def",
      "("
    ],
    "rationales_indexes": [
      33,
      39,
      43
    ],
    "token": "d"
  },
  "45": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.3366473317146301,
      0.04190976172685623,
      0.17280977964401245,
      0.0001753198157530278
    ],
    "rationales": [
      " add",
      "(",
      ",",
      "d"
    ],
    "rationales_indexes": [
      1,
      4,
      23,
      44
    ],
    "token": ","
  },
  "46": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      9.585198858985677e-05,
      0.07595914602279663,
      0.017720045521855354,
      0.14849483966827393,
      4.808398568201255e-09
    ],
    "rationales": [
      "def",
      ",",
      " key",
      "[",
      ","
    ],
    "rationales_indexes": [
      0,
      6,
      7,
      14,
      45
    ],
    "token": " key"
  },
  "47": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.00013447455421555787,
      0.01801484450697899,
      0.01575792394578457,
      0.0029503770638257265,
      0.0012873458908870816,
      0.005076363682746887,
      0.06267830729484558,
      0.010370878502726555,
      0.0184763353317976,
      0.028641514480113983,
      0.02734476700425148,
      0.017040155827999115,
      0.025914782658219337,
      0.00879679061472416,
      0.028712444007396698,
      0.01874607615172863,
      0.018937377259135246,
      0.012458214536309242,
      0.2465219348669052,
      0.007169406861066818,
      0.02512211911380291,
      2.986095528831334e-12
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count",
      "(",
      "d",
      "):",
      "[",
      ".",
      "get",
      "key",
      " +",
      " 1",
      "key",
      "\n",
      "def",
      "_",
      "count",
      "(",
      "d",
      ",",
      " key"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      8,
      14,
      19,
      20,
      22,
      26,
      27,
      35,
      38,
      39,
      41,
      42,
      43,
      44,
      45,
      46
    ],
    "token": "):"
  },
  "48": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.0007692198269069195,
      0.10361278057098389,
      0.004665991757065058,
      0.031242936849594116,
      0.05039535090327263,
      0.21919189393520355,
      0.00012227057595737278
    ],
    "rationales": [
      "d",
      " ",
      ")",
      " ",
      " ",
      " key",
      "):"
    ],
    "rationales_indexes": [
      5,
      10,
      25,
      29,
      30,
      46,
      47
    ],
    "token": " "
  },
  "49": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9995001554489136
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      48
    ],
    "token": " "
  },
  "5": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "With"
    ],
    "probabilities": [
      0.0034916435834020376,
      0.0055622318759560585,
      0.005976979620754719,
      0.004753738176077604,
      0.0001402340567437932
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "d"
  },
  "50": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9993146657943726
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      49
    ],
    "token": " "
  },
  "51": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.02977009303867817,
      0.25211137533187866,
      0.0006661651423200965,
      0.008914907462894917,
      0.0925828069448471,
      0.030501695349812508,
      5.372837676986819e-07
    ],
    "rationales": [
      "(",
      "d",
      "(",
      " d",
      "\n",
      "count",
      " "
    ],
    "rationales_indexes": [
      4,
      5,
      21,
      33,
      37,
      42,
      50
    ],
    "token": " d"
  },
  "52": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.09249714016914368,
      0.009426010772585869,
      0.006307332310825586,
      0.012677399441599846,
      0.021639877930283546,
      0.011330603621900082,
      0.020504865795373917,
      0.010880235582590103,
      0.020173048600554466,
      0.010977817699313164,
      0.004742664285004139,
      0.01968071423470974,
      0.007834059186279774,
      0.013179793953895569,
      0.462474524974823,
      0.006200618576258421,
      0.002220156602561474,
      0.017059149220585823,
      0.017968513071537018,
      0.019667383283376694,
      0.01912553608417511,
      0.023966293781995773,
      0.01678849570453167,
      0.012036334723234177,
      5.081375875626293e-10
    ],
    "rationales": [
      "def",
      "_",
      ",",
      " ",
      "[",
      "key",
      "]",
      " =",
      ".",
      "key",
      ",",
      " +",
      "\n",
      " ",
      " ",
      " d",
      "[",
      "key",
      "]",
      "\n",
      "_",
      " ",
      " ",
      " ",
      " d"
    ],
    "rationales_indexes": [
      0,
      2,
      6,
      10,
      14,
      15,
      16,
      17,
      19,
      22,
      23,
      26,
      28,
      29,
      31,
      33,
      34,
      35,
      36,
      37,
      41,
      48,
      49,
      50,
      51
    ],
    "token": "["
  },
  "53": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      2.6974028514814563e-05,
      0.03742688149213791,
      0.060214534401893616,
      0.013612566515803337,
      0.029342446476221085,
      0.02166496403515339,
      0.005686246324330568,
      0.0005591100198216736,
      8.811791758489562e-07
    ],
    "rationales": [
      "def",
      "count",
      " key",
      " return",
      "key",
      "_",
      "d",
      " key",
      "["
    ],
    "rationales_indexes": [
      0,
      3,
      7,
      32,
      35,
      41,
      44,
      46,
      52
    ],
    "token": "key"
  },
  "54": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.32846206426620483,
      1.1980111347398292e-10
    ],
    "rationales": [
      "[",
      "key"
    ],
    "rationales_indexes": [
      52,
      53
    ],
    "token": "]"
  },
  "6": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.01885780692100525,
      0.12781213223934174,
      0.033127184957265854,
      6.917683640494943e-05
    ],
    "rationales": [
      "def",
      " add",
      "(",
      "d"
    ],
    "rationales_indexes": [
      0,
      1,
      4,
      5
    ],
    "token": ","
  },
  "7": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.0001436206221114844,
      0.0009166409727185965,
      0.0015406198799610138,
      0.001228846376761794,
      0.0019073205767199397,
      0.001716524362564087,
      1.238252878010826e-07
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count",
      "(",
      "d",
      ","
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6
    ],
    "token": " key"
  },
  "8": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.00194158882368356,
      0.027000200003385544,
      0.057353146374225616,
      0.04940948262810707,
      0.012043284252285957,
      0.04022543504834175,
      0.002966556465253234,
      1.3961952254060161e-12
    ],
    "rationales": [
      "def",
      " add",
      "_",
      "count",
      "(",
      "d",
      ",",
      " key"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ],
    "token": "):"
  },
  "9": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Structural"
    ],
    "probabilities": [
      0.9328515529632568,
      0.05232931673526764
    ],
    "rationales": [
      "count",
      "):"
    ],
    "rationales_indexes": [
      3,
      8
    ],
    "token": "\n"
  },
  "_phrase": "def add_count(d, key):\n    d[key] = d.get(key, 0) + 1\n    return d[key]\n\ndef add_count(d, key):    d[key]"
}
curl -X POST http://127.0.0.1:5000/prompt -H "Content-Type: application/json"  0.02s user 0.02s system 0% cpu 42.632 total


Sample 5

time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def join_words(words):\n    return \",\".join(w.strip() for w in words if w)\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      3.359693073434755e-05
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " join"
  },
  "10": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.9994294047355652
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      9
    ],
    "token": " "
  },
  "11": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      3.0085380785749294e-05,
      0.022343488410115242,
      0.007862988859415054,
      0.037078212946653366,
      0.024887047708034515,
      0.03295378386974335,
      0.00048712926218286157,
      0.0001485444518039003,
      0.0018389546312391758,
      0.015742193907499313,
      2.459069614602072e-10
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words",
      "(",
      "words",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10
    ],
    "token": " return"
  },
  "12": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.017385907471179962,
      0.029666762799024582,
      0.07178346067667007,
      0.10219084471464157,
      0.05405666306614876,
      2.2694543986290228e-06
    ],
    "rationales": [
      "def",
      "words",
      "(",
      " ",
      " ",
      " return"
    ],
    "rationales_indexes": [
      0,
      3,
      4,
      9,
      10,
      11
    ],
    "token": " \""
  },
  "13": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      8.284715295303613e-05,
      0.008023669943213463,
      0.006756726652383804,
      0.007352862972766161,
      0.0007547052227891982,
      0.007723579648882151,
      0.0009455332765355706,
      0.0043885246850550175,
      0.006468184292316437,
      0.005845558363944292,
      0.005249041132628918,
      0.001568207866512239,
      0.00030494490056298673
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words",
      "(",
      "words",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " return",
      " \""
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12
    ],
    "token": ",\""
  },
  "14": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.014115925878286362,
      0.04688981920480728,
      0.028168223798274994,
      0.00027228298131376505
    ],
    "rationales": [
      "def",
      "):",
      "\n",
      ",\""
    ],
    "rationales_indexes": [
      0,
      6,
      7,
      13
    ],
    "token": "."
  },
  "15": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      2.9197122785262764e-05,
      0.07854273170232773,
      0.5955460667610168,
      2.661319498109549e-10
    ],
    "rationales": [
      "def",
      " join",
      " \"",
      "."
    ],
    "rationales_indexes": [
      0,
      1,
      12,
      14
    ],
    "token": "join"
  },
  "16": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.12315452843904495,
      0.0038372105918824673,
      0.45847088098526,
      2.8384769393596798e-05
    ],
    "rationales": [
      "def",
      "(",
      ".",
      "join"
    ],
    "rationales_indexes": [
      0,
      4,
      14,
      15
    ],
    "token": "("
  },
  "17": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0011745078954845667,
      0.002826917450875044,
      0.0018205501837655902,
      0.00238538789562881,
      0.002164531033486128,
      0.0023708788212388754,
      0.002488925587385893,
      0.002949576359242201,
      0.0027629591058939695,
      0.0028138370253145695,
      0.002797547960653901,
      0.0019918999169021845,
      0.0024721103254705667,
      0.0017259507440030575,
      0.001489653019234538,
      0.0017394130118191242,
      3.631928120739758e-05
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words",
      "(",
      "words",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " return",
      " \"",
      ",\"",
      ".",
      "join",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16
    ],
    "token": "w"
  },
  "18": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.08614146709442139,
      2.8202980502101127e-06
    ],
    "rationales": [
      "def",
      "w"
    ],
    "rationales_indexes": [
      0,
      17
    ],
    "token": "."
  },
  "19": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      4.966595497535309e-06,
      0.00023495152709074318,
      0.006302420049905777,
      0.00116064737085253,
      0.0021344060078263283,
      0.003199997590854764,
      0.005026528146117926,
      0.006199193652719259,
      0.011756068095564842,
      0.011423454619944096,
      0.0037563107907772064,
      0.002911082236096263,
      0.008229491300880909,
      0.009039975702762604,
      0.004435047507286072,
      0.006940765772014856,
      0.003931623417884111,
      0.011798759922385216,
      1.4399889058935855e-09
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words",
      "(",
      "words",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " return",
      " \"",
      ",\"",
      ".",
      "join",
      "(",
      "w",
      "."
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18
    ],
    "token": "strip"
  },
  "2": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.06020958721637726,
      1.7502506111100047e-08
    ],
    "rationales": [
      "def",
      " join"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "_"
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.013289717957377434,
      0.14909707009792328,
      2.1489845458688706e-09
    ],
    "rationales": [
      "def",
      ".",
      "strip"
    ],
    "rationales_indexes": [
      0,
      18,
      19
    ],
    "token": "()"
  },
  "21": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.00813065655529499,
      0.1849680095911026,
      0.0009467797353863716,
      0.004196732770651579,
      0.047421809285879135,
      0.001858394593000412,
      1.1236064665354206e-06
    ],
    "rationales": [
      "def",
      "):",
      " return",
      "(",
      ".",
      "strip",
      "()"
    ],
    "rationales_indexes": [
      0,
      6,
      11,
      16,
      18,
      19,
      20
    ],
    "token": " for"
  },
  "22": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.1290038377046585,
      0.05611301213502884,
      0.05022692680358887,
      0.054181963205337524,
      0.04535853862762451,
      0.0013942805817350745,
      0.23493526875972748,
      0.19642630219459534,
      3.8918646168895066e-05
    ],
    "rationales": [
      "def",
      "\n",
      " ",
      " ",
      " return",
      "w",
      ".",
      "strip",
      " for"
    ],
    "rationales_indexes": [
      0,
      7,
      9,
      10,
      11,
      17,
      18,
      19,
      21
    ],
    "token": " w"
  },
  "23": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0673697218298912,
      0.5238060355186462,
      0.006914983503520489,
      0.018539533019065857,
      0.03630644083023071,
      0.028692565858364105,
      0.018211299553513527,
      0.0023149969056248665,
      1.9768836523326172e-07
    ],
    "rationales": [
      "def",
      "):",
      "join",
      "w",
      ".",
      "strip",
      "()",
      " for",
      " w"
    ],
    "rationales_indexes": [
      0,
      6,
      15,
      17,
      18,
      19,
      20,
      21,
      22
    ],
    "token": " in"
  },
  "24": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.00018871287466026843,
      0.0033475318923592567,
      0.01932620070874691,
      0.005588827189058065,
      0.011163974180817604,
      0.0316305086016655,
      0.03658575564622879,
      0.0695853903889656,
      0.19777356088161469,
      2.5013011963892495e-06
    ],
    "rationales": [
      "def",
      "words",
      "(",
      "words",
      "\n",
      " ",
      "(",
      " for",
      " w",
      " in"
    ],
    "rationales_indexes": [
      0,
      3,
      4,
      5,
      7,
      8,
      16,
      21,
      22,
      23
    ],
    "token": " words"
  },
  "25": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0038858246989548206,
      0.00504264235496521,
      0.014702029526233673,
      0.014990627765655518,
      0.01948774978518486,
      0.016575735062360764,
      0.004702644422650337,
      0.026191161945462227,
      0.048046357929706573,
      0.040539227426052094,
      0.03183264657855034,
      0.012867085635662079,
      0.03363432735204697,
      0.04093640670180321,
      0.0360981710255146,
      0.029111433774232864,
      0.005351305939257145,
      0.029779164120554924,
      0.030355380848050117,
      0.016940660774707794,
      0.01879119873046875,
      0.01991637609899044,
      0.013834001496434212,
      0.01984640397131443,
      8.807417088974034e-07
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words",
      "(",
      "words",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " return",
      " \"",
      ",\"",
      ".",
      "join",
      "(",
      "w",
      ".",
      "strip",
      "()",
      " for",
      " w",
      " in",
      " words"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24
    ],
    "token": " if"
  },
  "26": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.05190107226371765,
      0.789603590965271,
      1.6482405044371262e-05
    ],
    "rationales": [
      " w",
      " words",
      " if"
    ],
    "rationales_indexes": [
      22,
      24,
      25
    ],
    "token": " w"
  },
  "27": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.05588863790035248,
      0.30859148502349854,
      3.189187358643153e-09
    ],
    "rationales": [
      "def",
      "(",
      " w"
    ],
    "rationales_indexes": [
      0,
      4,
      26
    ],
    "token": ")"
  },
  "28": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.754012405872345,
      0.062161702662706375
    ],
    "rationales": [
      " return",
      ")"
    ],
    "rationales_indexes": [
      11,
      27
    ],
    "token": "\n"
  },
  "29": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.995965838432312
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      28
    ],
    "token": "\n"
  },
  "3": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      8.434872870566323e-05,
      0.0006725810235366225,
      8.090543346384038e-09
    ],
    "rationales": [
      "def",
      " join",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "words"
  },
  "30": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.001736163510940969,
      0.10564636439085007,
      0.027777153998613358,
      1.6909720290669839e-09
    ],
    "rationales": [
      "def",
      "words",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      3,
      6,
      29
    ],
    "token": "def"
  },
  "31": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Statements"
    ],
    "probabilities": [
      9.754400707606692e-06,
      0.05311952531337738,
      0.4254857897758484,
      1.2493331924545714e-09
    ],
    "rationales": [
      "def",
      " join",
      "\n",
      "def"
    ],
    "rationales_indexes": [
      0,
      1,
      29,
      30
    ],
    "token": " join"
  },
  "32": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "With"
    ],
    "probabilities": [
      0.11973968893289566,
      0.026920471340417862,
      2.1160485630389303e-07
    ],
    "rationales": [
      "def",
      "_",
      " join"
    ],
    "rationales_indexes": [
      0,
      2,
      31
    ],
    "token": "_"
  },
  "33": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "With"
    ],
    "probabilities": [
      0.0024629635736346245,
      0.0003375087399035692,
      0.00012049816723447293,
      0.03032129444181919,
      6.320941992044027e-08
    ],
    "rationales": [
      "def",
      "words",
      " words",
      " join",
      "_"
    ],
    "rationales_indexes": [
      0,
      3,
      24,
      31,
      32
    ],
    "token": "words"
  },
  "34": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Types"
    ],
    "probabilities": [
      0.06261228770017624,
      0.1774040013551712,
      0.10652659833431244,
      0.19465631246566772,
      0.016980407759547234,
      2.2313950012176065e-06
    ],
    "rationales": [
      "def",
      "):",
      " if",
      " w",
      ")",
      "words"
    ],
    "rationales_indexes": [
      0,
      6,
      25,
      26,
      27,
      33
    ],
    "token": "("
  },
  "35": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      1.9607095964602195e-05,
      0.10214337706565857,
      0.0013883678475394845,
      0.25452619791030884,
      6.084692927288415e-08
    ],
    "rationales": [
      "def",
      "(",
      "words",
      "):",
      "("
    ],
    "rationales_indexes": [
      0,
      4,
      5,
      6,
      34
    ],
    "token": "words"
  },
  "36": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.0005454714992083609,
      0.011414064094424248,
      0.017022404819726944,
      0.00662859994918108,
      0.05468015745282173,
      0.016888340935111046,
      0.013410680927336216,
      0.015029198490083218,
      0.01945681869983673,
      0.19886799156665802,
      0.06978057324886322,
      0.12150512635707855,
      0.09147914499044418,
      2.6398125907434178e-09
    ],
    "rationales": [
      "def",
      "_",
      "words",
      "(",
      "):",
      "\n",
      ".",
      "(",
      "w",
      "()",
      " for",
      " in",
      "def",
      "words"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      4,
      6,
      7,
      14,
      16,
      17,
      20,
      21,
      23,
      30,
      35
    ],
    "token": "):"
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.896772027015686,
      0.08076635003089905
    ],
    "rationales": [
      " \"",
      "):"
    ],
    "rationales_indexes": [
      12,
      36
    ],
    "token": "\n"
  },
  "38": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9953950047492981
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      37
    ],
    "token": "\n"
  },
  "39": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.013593805953860283,
      0.059338267892599106,
      5.56532222617534e-06
    ],
    "rationales": [
      "def",
      "_",
      "\n"
    ],
    "rationales_indexes": [
      0,
      2,
      38
    ],
    "token": "#"
  },
  "4": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.05193426087498665,
      0.029563359916210175,
      0.03510073944926262,
      1.8146110960515216e-06
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "("
  },
  "40": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.0005906145088374615,
      0.05545548349618912,
      0.013371647335588932,
      0.047217849642038345,
      0.044180333614349365,
      0.04808729887008667,
      0.020942555740475655,
      0.03499531000852585,
      0.02862447500228882,
      0.006604004185646772,
      0.0397733673453331,
      0.001189706614241004,
      4.5347437094278575e-07
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "\n",
      " return",
      " \"",
      "def",
      "(",
      "words",
      "):",
      "\n",
      "\n",
      "#"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      7,
      11,
      12,
      30,
      34,
      35,
      36,
      37,
      38,
      39
    ],
    "token": " This"
  },
  "41": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.053790051490068436,
      0.05999715253710747,
      0.08077336102724075,
      0.0884702131152153,
      0.003425931092351675
    ],
    "rationales": [
      "def",
      ",\"",
      " for",
      " in",
      " This"
    ],
    "rationales_indexes": [
      0,
      13,
      21,
      23,
      40
    ],
    "token": " is"
  },
  "42": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Statements"
    ],
    "probabilities": [
      0.07195357978343964,
      0.06523413956165314,
      0.05645744130015373,
      0.06332384794950485,
      0.05746786296367645,
      0.036940693855285645,
      0.13920560479164124,
      0.06457490473985672,
      0.05758075416088104,
      0.00975959375500679
    ],
    "rationales": [
      "def",
      ",\"",
      " words",
      " if",
      " w",
      "):",
      "\n",
      "#",
      " This",
      " is"
    ],
    "rationales_indexes": [
      0,
      13,
      24,
      25,
      26,
      36,
      38,
      39,
      40,
      41
    ],
    "token": " the"
  },
  "43": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.05568801611661911,
      0.016379663720726967,
      0.002895655808970332,
      0.037451744079589844,
      0.03584618121385574,
      0.032610781490802765,
      0.03842814266681671,
      0.0050178952515125275,
      0.039271797984838486,
      0.005693898070603609,
      0.006469495594501495,
      0.046531904488801956,
      0.005618855357170105,
      0.04513053968548775,
      0.04505274444818497,
      0.04373711720108986,
      0.03813248127698898,
      0.005693886429071426,
      0.11305146664381027,
      0.022884340956807137,
      0.030368201434612274,
      0.03509235009551048,
      0.011493243277072906,
      0.04388527199625969,
      0.042147353291511536,
      4.408856923987514e-08
    ],
    "rationales": [
      "def",
      " join",
      "(",
      "words",
      "\n",
      " ",
      " ",
      " return",
      ",\"",
      "join",
      "(",
      "w",
      ".",
      "strip",
      "()",
      " for",
      " w",
      " in",
      " words",
      " if",
      " w",
      "\n",
      "\n",
      "words",
      " is",
      " the"
    ],
    "rationales_indexes": [
      0,
      1,
      4,
      5,
      7,
      8,
      9,
      11,
      13,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      28,
      29,
      35,
      41,
      42
    ],
    "token": " same"
  },
  "44": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.22698034346103668,
      0.04331028088927269,
      3.170659329043701e-05
    ],
    "rationales": [
      "words",
      " is",
      " same"
    ],
    "rationales_indexes": [
      35,
      41,
      43
    ],
    "token": " as"
  },
  "45": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.010961424559354782,
      0.26701706647872925,
      0.11750701069831848,
      2.7288544515613467e-05
    ],
    "rationales": [
      "(",
      " is",
      " the",
      " as"
    ],
    "rationales_indexes": [
      34,
      41,
      42,
      44
    ],
    "token": " the"
  },
  "46": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.7825117111206055,
      0.010173904709517956,
      3.2210255795916964e-08
    ],
    "rationales": [
      " w",
      " join",
      " the"
    ],
    "rationales_indexes": [
      26,
      31,
      45
    ],
    "token": " join"
  },
  "47": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.10909979045391083,
      0.5952962040901184,
      0.025892049074172974,
      2.1208516898241214e-07
    ],
    "rationales": [
      "def",
      " join",
      "_",
      " join"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      46
    ],
    "token": "_"
  },
  "48": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.002357009332627058,
      0.05213092267513275,
      0.007912199944257736,
      0.021600835025310516,
      0.006424552295356989,
      0.0005662059411406517,
      0.4154026508331299,
      6.890608972298651e-08
    ],
    "rationales": [
      "def",
      "_",
      "words",
      "words",
      "words",
      "words",
      " join",
      "_"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      5,
      33,
      35,
      46,
      47
    ],
    "token": "words"
  },
  "49": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Bool"
    ],
    "probabilities": [
      0.0005558867123909295,
      0.0191040001809597,
      0.010261434130370617,
      0.0030446432065218687,
      0.09246378391981125,
      0.17086966335773468,
      0.052973151206970215,
      0.030014367774128914,
      0.20745214819908142,
      0.12133342772722244,
      8.431365827732407e-09
    ],
    "rationales": [
      "def",
      "_",
      "):",
      " return",
      ".",
      "join",
      "strip",
      "()",
      " the",
      "_",
      "words"
    ],
    "rationales_indexes": [
      0,
      2,
      6,
      11,
      14,
      15,
      19,
      20,
      42,
      47,
      48
    ],
    "token": "()"
  },
  "5": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Bool"
    ],
    "probabilities": [
      3.5975601349491626e-05,
      0.011630792170763016,
      0.011152524501085281,
      0.005053674802184105,
      9.366315367742573e-08
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "words"
  },
  "50": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.00919217336922884,
      0.006932399235665798,
      0.005113007966428995,
      0.006455956492573023,
      0.009981145150959492,
      0.010636620223522186,
      0.007636140566319227,
      0.021695872768759727,
      0.0056372033432126045,
      0.006653910502791405,
      0.0063382345251739025,
      0.12327445298433304,
      0.006862638518214226,
      0.004772766958922148,
      0.1903497725725174,
      0.06321610510349274,
      1.631802115298342e-05
    ],
    "rationales": [
      "def",
      " return",
      " if",
      "\n",
      "def",
      "words",
      "words",
      "):",
      "\n",
      "#",
      " This",
      " is",
      " the",
      " same",
      " the",
      "words",
      "()"
    ],
    "rationales_indexes": [
      0,
      11,
      25,
      28,
      30,
      33,
      35,
      36,
      38,
      39,
      40,
      41,
      42,
      43,
      45,
      48,
      49
    ],
    "token": " method"
  },
  "51": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      0.5478108525276184,
      0.101340651512146,
      0.10117631405591965,
      0.00011475240171421319
    ],
    "rationales": [
      ",\"",
      "(",
      "\n",
      " method"
    ],
    "rationales_indexes": [
      13,
      34,
      38,
      50
    ],
    "token": ","
  },
  "52": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      8.975938544608653e-05,
      0.022638866677880287,
      0.00566819217056036,
      0.008729923516511917,
      0.02362864837050438,
      0.003095156978815794,
      0.0227230004966259,
      0.0021249183919280767,
      0.020581314340233803,
      0.013515953905880451,
      0.27519842982292175,
      0.000642170780338347,
      0.023006724193692207,
      0.17051135003566742,
      0.24120856821537018,
      0.09105667471885681,
      0.05809381231665611,
      0.02529025264084339,
      0.32545971870422363,
      0.4353023171424866,
      0.001425430178642273,
      0.02500225603580475,
      0.1278064101934433,
      0.1951788365840912,
      8.848986254861302e-08
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words",
      "words",
      "):",
      "\n",
      " return",
      "()",
      " for",
      " w",
      " if",
      ")",
      "def",
      " join",
      "words",
      "words",
      " This",
      " is",
      " the",
      " same",
      " as",
      "_",
      "()",
      ","
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      5,
      6,
      7,
      11,
      20,
      21,
      22,
      25,
      27,
      30,
      31,
      33,
      35,
      40,
      41,
      42,
      43,
      44,
      47,
      49,
      51
    ],
    "token": " except"
  },
  "53": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.09397806227207184,
      0.00034169561695307493
    ],
    "rationales": [
      "def",
      " except"
    ],
    "rationales_indexes": [
      0,
      52
    ],
    "token": " that"
  },
  "54": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.27686476707458496,
      0.05392903834581375,
      0.046496178954839706,
      0.027727311477065086,
      0.03569116070866585,
      0.0430116169154644,
      0.04432464390993118,
      0.044343337416648865,
      0.24370400607585907,
      0.027566296979784966,
      0.06844362616539001,
      0.03916556015610695,
      0.1616387516260147,
      0.04070692136883736,
      0.0960417091846466,
      0.04116666689515114,
      0.023683378472924232,
      0.15127559006214142,
      0.1286342591047287,
      0.13624192774295807,
      0.12022656947374344,
      0.26068350672721863,
      0.03232119604945183,
      0.10804153978824615,
      0.024430308490991592,
      0.2579752206802368,
      0.12878115475177765,
      0.03023427538573742,
      0.20474641025066376,
      0.17558453977108002,
      0.17113609611988068,
      0.13473714888095856,
      0.13335128128528595,
      0.14808760583400726,
      0.05014188215136528,
      0.0018458003178238869
    ],
    "rationales": [
      "def",
      "_",
      "words",
      " ",
      " ",
      " ",
      " return",
      " \"",
      ".",
      "join",
      "(",
      "strip",
      "()",
      " for",
      " w",
      " in",
      " words",
      "\n",
      "\n",
      "def",
      " join",
      "_",
      "words",
      "(",
      "words",
      "\n",
      "#",
      " This",
      " the",
      " as",
      "_",
      "words",
      "()",
      " method",
      " except",
      " that"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      8,
      9,
      10,
      11,
      12,
      14,
      15,
      16,
      19,
      20,
      21,
      22,
      23,
      24,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      37,
      39,
      40,
      42,
      44,
      47,
      48,
      49,
      50,
      52,
      53
    ],
    "token": " it"
  },
  "6": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.0011497382074594498,
      0.011028026230633259,
      0.009814141318202019,
      0.014466067776083946,
      0.004595932085067034,
      2.6583075740660433e-09
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words",
      "(",
      "words"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "token": "):"
  },
  "7": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.16685034334659576,
      0.06029180809855461
    ],
    "rationales": [
      "def",
      "):"
    ],
    "rationales_indexes": [
      0,
      6
    ],
    "token": "\n"
  },
  "8": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      2.4317382667504717e-06,
      6.53448369121179e-05,
      3.039685543626547e-05,
      9.816662350203842e-05,
      1.69410377566237e-05,
      4.8289810365531594e-05,
      1.0863338502531406e-05,
      1.7774368643586058e-08
    ],
    "rationales": [
      "def",
      " join",
      "_",
      "words",
      "(",
      "words",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ],
    "token": " "
  },
  "9": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.999458372592926
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      8
    ],
    "token": " "
  },
  "_phrase": "def join_words(words):\n    return \",\".join(w.strip() for w in words if w)\n\ndef join_words(words):\n\n# This is the same as the join_words() method, except that it"
}
curl -X POST http://127.0.0.1:5000/prompt -H "Content-Type: application/json"  0.02s user 0.02s system 0% cpu 56.968 total


Sample 6

time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def pop_default(lst, default=None):\n    if not lst:\n        return default\n    return lst.pop()\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      2.542955735407304e-05
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " pop"
  },
  "10": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.00017566837777849287,
      0.08414003252983093,
      0.08460976183414459,
      0.12106088548898697,
      0.0709880143404007,
      0.012310917489230633,
      3.7213436776539766e-09
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "(",
      ",",
      " default",
      "="
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      4,
      7,
      8,
      9
    ],
    "token": "None"
  },
  "11": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      3.645218748715706e-05,
      0.020063579082489014,
      0.018410932272672653,
      0.02722581848502159,
      0.007955387234687805,
      0.025388255715370178,
      0.030066432431340218,
      0.002973722293972969,
      0.018795276060700417,
      0.016121963039040565,
      7.722027817180788e-07
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "default",
      "(",
      "l",
      "st",
      ",",
      " default",
      "=",
      "None"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10
    ],
    "token": "):"
  },
  "12": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.34196150302886963,
      0.056338485330343246
    ],
    "rationales": [
      "st",
      "):"
    ],
    "rationales_indexes": [
      6,
      11
    ],
    "token": "\n"
  },
  "13": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      1.60430101914244e-06,
      2.2436358904087683e-06,
      1.9095707557426067e-06,
      0.00014698010636493564,
      0.00018183857901021838,
      2.176351472371607e-06,
      2.1912519514444284e-06,
      0.00027619083994068205,
      3.61593401976279e-06,
      0.00013273701188154519,
      0.00021153503621462733,
      1.258102656720439e-05,
      2.5014429638758884e-08
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "default",
      "(",
      "l",
      "st",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12
    ],
    "token": " "
  },
  "14": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9995275735855103
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      13
    ],
    "token": " "
  },
  "15": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9994822144508362
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      14
    ],
    "token": " "
  },
  "16": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.0003448010829742998,
      0.0025660486426204443,
      0.0008113548392429948,
      0.005142410285770893,
      0.0003781118430197239,
      0.019102344289422035,
      0.017936794087290764,
      0.016250137239694595,
      0.01968478038907051,
      0.018758121877908707,
      0.015540840104222298,
      0.0016694857040420175,
      0.0022237820085138083,
      0.019789179787039757,
      0.018383294343948364,
      1.0330837341143706e-07
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "default",
      "(",
      "l",
      "st",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15
    ],
    "token": " if"
  },
  "17": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.04746171459555626,
      0.01390561182051897,
      0.06610233336687088,
      0.02188687212765217,
      0.07233715057373047,
      0.053847577422857285,
      0.03540883958339691,
      0.012908538803458214,
      0.022545142099261284,
      0.020281363278627396,
      0.023578528314828873,
      0.004238509573042393
    ],
    "rationales": [
      "def",
      " pop",
      "default",
      "(",
      ",",
      "=",
      "None",
      "\n",
      " ",
      " ",
      " ",
      " if"
    ],
    "rationales_indexes": [
      0,
      1,
      3,
      4,
      7,
      9,
      10,
      12,
      13,
      14,
      15,
      16
    ],
    "token": " not"
  },
  "18": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.013534262776374817,
      0.005137712694704533,
      0.01148284412920475,
      0.007802946493029594,
      0.02820974588394165,
      0.001090707490220666,
      0.0018418168183416128,
      0.016793137416243553,
      0.023205939680337906,
      0.019103821367025375,
      0.1194879412651062,
      8.570804311602842e-06
    ],
    "rationales": [
      "def",
      "(",
      "l",
      "st",
      " default",
      "None",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " not"
    ],
    "rationales_indexes": [
      0,
      4,
      5,
      6,
      8,
      10,
      12,
      13,
      14,
      15,
      16,
      17
    ],
    "token": " l"
  },
  "19": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.0004943765234202147,
      0.45615026354789734,
      0.027414757758378983,
      0.006657127756625414,
      7.903140364362571e-09
    ],
    "rationales": [
      "def",
      "(",
      "l",
      "st",
      " l"
    ],
    "rationales_indexes": [
      0,
      4,
      5,
      6,
      18
    ],
    "token": "st"
  },
  "2": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Statements"
    ],
    "probabilities": [
      0.07080712169408798,
      9.134835943225283e-11
    ],
    "rationales": [
      "def",
      " pop"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "_"
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.005591392517089844,
      0.06879109889268875,
      0.03434447571635246,
      0.05104457214474678,
      0.2891275882720947,
      0.08490891009569168,
      1.2837630492867902e-05
    ],
    "rationales": [
      "def",
      " pop",
      "):",
      " ",
      " if",
      " l",
      "st"
    ],
    "rationales_indexes": [
      0,
      1,
      11,
      13,
      16,
      18,
      19
    ],
    "token": ":"
  },
  "21": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Structural"
    ],
    "probabilities": [
      0.10067862272262573,
      0.12651585042476654,
      0.1467968374490738,
      0.1329067051410675,
      0.4818468689918518,
      0.12159201502799988,
      3.8325269997585565e-06
    ],
    "rationales": [
      "_",
      "st",
      "=",
      "):",
      " ",
      "st",
      ":"
    ],
    "rationales_indexes": [
      2,
      6,
      9,
      11,
      13,
      19,
      20
    ],
    "token": "\n"
  },
  "22": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.002214460400864482,
      0.576374351978302,
      0.01777430810034275,
      1.2532242976703856e-07
    ],
    "rationales": [
      " default",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      8,
      13,
      15,
      21
    ],
    "token": " "
  },
  "23": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.999372661113739
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      22
    ],
    "token": " "
  },
  "24": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "OOP"
    ],
    "probabilities": [
      0.9994456171989441
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      23
    ],
    "token": " "
  },
  "25": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.9993972778320312
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      24
    ],
    "token": " "
  },
  "26": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9993921518325806
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      25
    ],
    "token": " "
  },
  "27": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9993649125099182
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      26
    ],
    "token": " "
  },
  "28": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9993835687637329
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      27
    ],
    "token": " "
  },
  "29": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      2.229094570793677e-05,
      0.015367438085377216,
      0.0051170410588383675,
      0.02397213876247406,
      0.03624679520726204,
      0.0006204293458722532,
      7.498219267709771e-10
    ],
    "rationales": [
      "def",
      "default",
      "):",
      "\n",
      " ",
      " if",
      " "
    ],
    "rationales_indexes": [
      0,
      3,
      11,
      12,
      13,
      16,
      28
    ],
    "token": " return"
  },
  "3": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.0021892348304390907,
      0.0012155055301263928,
      2.518999906442332e-07
    ],
    "rationales": [
      "def",
      " pop",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "default"
  },
  "30": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Types"
    ],
    "probabilities": [
      0.2887107729911804,
      0.158229798078537,
      0.008911889046430588,
      0.03298390284180641,
      0.07845154404640198,
      0.06742602586746216,
      0.08481577783823013,
      0.0701608955860138,
      0.07303513586521149,
      0.07897385954856873,
      0.0730535238981247,
      4.388329398352653e-05
    ],
    "rationales": [
      "default",
      " default",
      "\n",
      " ",
      " ",
      " not",
      " l",
      "st",
      " ",
      " ",
      " ",
      " return"
    ],
    "rationales_indexes": [
      3,
      8,
      12,
      13,
      14,
      17,
      18,
      19,
      22,
      23,
      24,
      29
    ],
    "token": " default"
  },
  "31": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Statements"
    ],
    "probabilities": [
      0.9370458722114563,
      1.625935475146889e-08
    ],
    "rationales": [
      "\n",
      " default"
    ],
    "rationales_indexes": [
      21,
      30
    ],
    "token": "\n"
  },
  "32": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.007033433299511671,
      0.006452900357544422,
      0.3408813774585724,
      3.775078880607907e-07
    ],
    "rationales": [
      "st",
      " default",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      6,
      8,
      28,
      31
    ],
    "token": " "
  },
  "33": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9994258880615234
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      32
    ],
    "token": " "
  },
  "34": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Asserts"
    ],
    "probabilities": [
      0.9992619156837463
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      33
    ],
    "token": " "
  },
  "35": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9024081230163574,
      4.870839287463014e-10
    ],
    "rationales": [
      " return",
      " "
    ],
    "rationales_indexes": [
      29,
      34
    ],
    "token": " return"
  },
  "36": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.0015343725681304932,
      0.0830933153629303,
      0.03448250889778137,
      0.009660695679485798,
      1.3954841904251225e-07
    ],
    "rationales": [
      "def",
      "):",
      " if",
      " l",
      " return"
    ],
    "rationales_indexes": [
      0,
      11,
      16,
      18,
      35
    ],
    "token": " l"
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00022231451293919235,
      0.27182841300964355,
      0.01128304935991764,
      0.0026074755005538464,
      1.1614976536122867e-08
    ],
    "rationales": [
      "def",
      "(",
      "l",
      "st",
      " l"
    ],
    "rationales_indexes": [
      0,
      4,
      5,
      6,
      36
    ],
    "token": "st"
  },
  "38": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.05758889764547348,
      0.07611937075853348,
      0.056808456778526306,
      0.0577675960958004,
      0.06113630160689354,
      0.04482442885637283,
      0.000138919785968028
    ],
    "rationales": [
      "def",
      "default",
      "l",
      " if",
      " not",
      "st",
      "st"
    ],
    "rationales_indexes": [
      0,
      3,
      5,
      16,
      17,
      19,
      37
    ],
    "token": "."
  },
  "39": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      7.052919499983545e-06,
      0.00286512216553092,
      0.17593888938426971,
      0.061125583946704865,
      1.2345672928404383e-07
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      ",",
      "."
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      7,
      38
    ],
    "token": "pop"
  },
  "4": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.008035183884203434,
      0.015441371127963066,
      0.013892486691474915,
      1.0651976189990364e-08
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "default"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "("
  },
  "40": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.00019664724823087454,
      0.07463811337947845,
      0.13615261018276215,
      0.14765681326389313,
      0.029902033507823944,
      0.004338387865573168,
      0.15386132895946503,
      0.1370130032300949,
      0.23371900618076324,
      0.13721396028995514,
      8.975502979735595e-11
    ],
    "rationales": [
      "def",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n",
      " return",
      " ",
      ".",
      "pop"
    ],
    "rationales_indexes": [
      0,
      7,
      8,
      9,
      10,
      11,
      12,
      29,
      34,
      38,
      39
    ],
    "token": "()"
  },
  "41": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.9452587366104126,
      5.4839314543642104e-05
    ],
    "rationales": [
      "\n",
      "()"
    ],
    "rationales_indexes": [
      21,
      40
    ],
    "token": "\n"
  },
  "42": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.9954813718795776
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      41
    ],
    "token": "\n"
  },
  "43": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.012366375885903835,
      0.05442896485328674,
      6.522656804008875e-06
    ],
    "rationales": [
      "def",
      "_",
      "\n"
    ],
    "rationales_indexes": [
      0,
      2,
      42
    ],
    "token": "#"
  },
  "44": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.0017238680738955736,
      0.03172141686081886,
      0.16178667545318604,
      8.796496331342496e-06
    ],
    "rationales": [
      "(",
      " if",
      "\n",
      "#"
    ],
    "rationales_indexes": [
      4,
      16,
      21,
      43
    ],
    "token": " if"
  },
  "45": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      8.496072950947564e-06,
      0.04106610640883446,
      0.006956952158361673,
      0.08226083964109421,
      0.15178373456001282,
      0.0023090720642358065,
      6.904588371980935e-05,
      0.0004536606720648706,
      1.5867366087718437e-08
    ],
    "rationales": [
      "def",
      "default",
      "):",
      ".",
      "pop",
      "\n",
      "\n",
      "#",
      " if"
    ],
    "rationales_indexes": [
      0,
      3,
      11,
      38,
      39,
      41,
      42,
      43,
      44
    ],
    "token": "def"
  },
  "46": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.0010743188904598355,
      0.0043565803207457066,
      0.16963031888008118,
      0.04437148943543434,
      0.14396171271800995,
      0.032252274453639984,
      0.024703092873096466,
      0.025596166029572487,
      0.017242969945073128,
      0.15980127453804016,
      0.04032839834690094,
      0.039652835577726364,
      0.010047675110399723,
      0.08543457835912704,
      0.023721447214484215,
      0.036442264914512634,
      0.03732752054929733,
      0.029927363619208336,
      0.04375958442687988,
      0.029526542872190475,
      0.1605496108531952,
      0.1728932112455368,
      0.12703454494476318,
      0.10945310443639755,
      0.05813254043459892,
      0.03921150043606758,
      0.18146705627441406,
      0.1528477966785431,
      0.2266615480184555,
      0.1590007245540619,
      3.5734089465222496e-08
    ],
    "rationales": [
      "def",
      "_",
      "l",
      "st",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " not",
      "st",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      "\n",
      "()",
      "\n",
      "#",
      " if",
      "def"
    ],
    "rationales_indexes": [
      0,
      2,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      31,
      40,
      41,
      43,
      44,
      45
    ],
    "token": " __"
  },
  "47": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      7.716585969319567e-05,
      0.011283894069492817,
      0.0016641400288790464,
      0.007182487286627293,
      0.0003108900273218751,
      0.004142275080084801,
      0.048367373645305634,
      0.15612207353115082,
      0.3285638988018036,
      0.038482069969177246,
      0.02976514957845211,
      0.19508425891399384,
      0.0928550437092781,
      0.2689027488231659,
      0.06091379001736641,
      0.020164689049124718,
      0.0006510134553536773,
      2.911067775723808e-10
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "l",
      "st",
      "):",
      "st",
      " ",
      " ",
      " return",
      " l",
      "st",
      "pop",
      "()",
      "\n",
      "#",
      " if",
      " __"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      5,
      6,
      11,
      19,
      33,
      34,
      35,
      36,
      37,
      39,
      40,
      42,
      43,
      44,
      46
    ],
    "token": "c"
  },
  "48": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.00027341730310581625,
      0.00019548642740119249,
      0.00019960623467341065,
      0.3895954191684723,
      0.00021913080126978457,
      0.08318916708230972,
      0.01877199299633503,
      1.2094027624698356e-05,
      0.00226771947927773,
      0.00015438023547176272,
      3.6249393815523945e-06,
      2.3597817033760293e-08
    ],
    "rationales": [
      "def",
      " return",
      " return",
      ".",
      "pop",
      "\n",
      "\n",
      "#",
      " if",
      "def",
      " __",
      "c"
    ],
    "rationales_indexes": [
      0,
      29,
      35,
      38,
      39,
      41,
      42,
      43,
      44,
      45,
      46,
      47
    ],
    "token": "plus"
  },
  "49": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9881474375724792
    ],
    "rationales": [
      "plus"
    ],
    "rationales_indexes": [
      48
    ],
    "token": "plus"
  },
  "5": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.0016827258514240384,
      0.0029953306075185537,
      0.002900016028434038,
      0.002450451022014022,
      0.00022861349862068892
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "default",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "l"
  },
  "50": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.7453866600990295,
      1.0701474820962176e-05
    ],
    "rationales": [
      "\n",
      "plus"
    ],
    "rationales_indexes": [
      41,
      49
    ],
    "token": "\n"
  },
  "51": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9972905516624451
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      50
    ],
    "token": "\n"
  },
  "52": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.047917481511831284,
      0.15721352398395538,
      0.18069568276405334,
      0.14915113151073456,
      0.38823530077934265,
      1.324971981375711e-05
    ],
    "rationales": [
      "#",
      "def",
      "c",
      "plus",
      "plus",
      "\n"
    ],
    "rationales_indexes": [
      43,
      45,
      47,
      48,
      49,
      51
    ],
    "token": "#"
  },
  "53": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      1.2940439773956314e-05,
      0.0028500754851847887,
      0.014833555556833744,
      0.04145638644695282,
      0.00150451494846493,
      0.17017287015914917,
      0.06583932042121887,
      0.04972591996192932,
      0.0005949039477854967,
      0.08275437355041504,
      0.03109900653362274,
      0.021276351064443588,
      0.11419451981782913,
      0.008795129135251045,
      1.0439551800800473e-07
    ],
    "rationales": [
      "def",
      "_",
      "default",
      "l",
      " default",
      " ",
      " ",
      "\n",
      " if",
      " __",
      "c",
      "plus",
      "plus",
      "\n",
      "#"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      5,
      30,
      33,
      34,
      42,
      44,
      46,
      47,
      48,
      49,
      51,
      52
    ],
    "token": " define"
  },
  "54": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.009151791222393513,
      0.1186017394065857,
      0.06185014545917511,
      0.35179388523101807,
      0.0516793467104435,
      0.1378365457057953,
      0.21587695181369781,
      0.06840832531452179,
      0.10787896066904068,
      8.244626314990455e-09
    ],
    "rationales": [
      "_",
      "\n",
      "\n",
      " default",
      "\n",
      "st",
      " if",
      " __",
      "\n",
      " define"
    ],
    "rationales_indexes": [
      2,
      12,
      21,
      30,
      31,
      37,
      44,
      46,
      51,
      53
    ],
    "token": " __"
  },
  "6": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.0028463457711040974,
      0.02519223652780056,
      0.022993501275777817,
      0.016797799617052078,
      0.008948090486228466,
      6.453789751503791e-07
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "default",
      "(",
      "l"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "token": "st"
  },
  "7": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.015083199366927147,
      0.12662887573242188,
      0.07375218719244003,
      0.030594132840633392,
      0.001171591691672802
    ],
    "rationales": [
      "def",
      " pop",
      "(",
      "l",
      "st"
    ],
    "rationales_indexes": [
      0,
      1,
      4,
      5,
      6
    ],
    "token": ","
  },
  "8": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.011801916174590588,
      0.011565269902348518,
      0.027308369055390358,
      0.006846924778074026,
      0.006329374387860298,
      0.02723659947514534,
      0.022367199882864952,
      3.0310695819935063e-07
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "default",
      "(",
      "l",
      "st",
      ","
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ],
    "token": " default"
  },
  "9": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.0052952333353459835,
      0.059349142014980316,
      0.049647796899080276,
      0.0585353784263134,
      0.05289767310023308,
      0.05600573122501373,
      0.06093325465917587,
      0.01882289908826351,
      6.065669494459414e-11
    ],
    "rationales": [
      "def",
      " pop",
      "_",
      "default",
      "(",
      "l",
      "st",
      ",",
      " default"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    "token": "="
  },
  "_phrase": "def pop_default(lst, default=None):\n    if not lst:\n        return default\n    return lst.pop()\n\n# ifdef __cplusplus\n\n# define __"
}
curl -X POST http://127.0.0.1:5000/prompt -H "Content-Type: application/json"  0.02s user 0.02s system 0% cpu 37.687 total



Sample 7

time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def is_pal(s):\n    t = \"\".join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.004530841019004583
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " is"
  },
  "10": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.9994294047355652
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      9
    ],
    "token": " "
  },
  "11": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.00034281489206478,
      0.004339989740401506,
      0.003909033723175526,
      0.006507610436528921,
      0.0010383722838014364,
      0.003645553020760417,
      0.0042084199376404285,
      0.0023619066923856735,
      0.004197718575596809,
      0.0008580596186220646,
      4.3812482886096404e-07
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10
    ],
    "token": " t"
  },
  "12": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.11560873687267303,
      3.6373585032833944e-08
    ],
    "rationales": [
      "def",
      " t"
    ],
    "rationales_indexes": [
      0,
      11
    ],
    "token": " ="
  },
  "13": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.0361565500497818,
      0.1879449486732483,
      0.006355843972414732
    ],
    "rationales": [
      "def",
      "pal",
      " ="
    ],
    "rationales_indexes": [
      0,
      3,
      12
    ],
    "token": " \""
  },
  "14": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      9.75687216850929e-05,
      0.0005189691437408328,
      0.005781047511845827,
      0.005089401733130217,
      0.0004945668042637408,
      0.0066589415073394775,
      0.007018405478447676,
      0.002507978118956089,
      0.003424019319936633,
      0.0036187772639095783,
      0.003977376967668533,
      0.004593416582792997,
      0.0013550196308642626,
      8.502371429131017e-07
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \""
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13
    ],
    "token": "\"."
  },
  "15": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      8.12458893051371e-05,
      0.019600261002779007,
      0.008554327301681042,
      0.0014847653219476342,
      0.16437934339046478,
      6.188908413529148e-11
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "\n",
      " \"",
      "\"."
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      7,
      13,
      14
    ],
    "token": "join"
  },
  "16": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.12315452843904495,
      0.0038372105918824673,
      0.6436682939529419,
      2.8384769393596798e-05
    ],
    "rationales": [
      "def",
      "(",
      "\".",
      "join"
    ],
    "rationales_indexes": [
      0,
      4,
      14,
      15
    ],
    "token": "("
  },
  "17": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.006793166045099497,
      0.02943367324769497,
      0.06360579282045364,
      0.0535128153860569,
      0.02936038374900818,
      0.02958093211054802,
      0.03536364808678627,
      0.043781861662864685,
      0.04504987597465515,
      0.06631950289011002,
      0.018425794318318367,
      0.010552001185715199,
      0.0433516651391983,
      0.038895804435014725,
      0.009526390582323074,
      0.01942652463912964,
      6.274571205722168e-05
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \"",
      "\".",
      "join",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16
    ],
    "token": "c"
  },
  "18": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.10326432436704636,
      7.993237522896379e-05
    ],
    "rationales": [
      "def",
      "c"
    ],
    "rationales_indexes": [
      0,
      17
    ],
    "token": "."
  },
  "19": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      4.209326561976923e-06,
      0.021926432847976685,
      0.000918956589885056,
      0.023777494207024574,
      0.021385904401540756,
      0.024314571171998978,
      0.0035855071619153023,
      0.023766962811350822,
      0.025005154311656952,
      0.02402527630329132,
      0.022664973512291908,
      0.015466510318219662,
      0.011349668726325035,
      0.020437536761164665,
      0.007627310696989298,
      0.0022818204015493393,
      0.00011337226169416681,
      0.000527530035469681,
      2.0235481346109196e-10
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \"",
      "\".",
      "join",
      "(",
      "c",
      "."
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18
    ],
    "token": "lower"
  },
  "2": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0032916837371885777,
      1.0387594073790751e-07
    ],
    "rationales": [
      "def",
      " is"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "_"
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.04282962530851364,
      0.08744484931230545,
      0.32381850481033325,
      9.469574990816909e-08
    ],
    "rationales": [
      "def",
      " is",
      ".",
      "lower"
    ],
    "rationales_indexes": [
      0,
      1,
      18,
      19
    ],
    "token": "()"
  },
  "21": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.00940045714378357,
      0.01868729665875435,
      0.0012491157976910472,
      0.060790423303842545,
      0.018671058118343353,
      0.014611788094043732,
      0.01702701300382614,
      0.019487032666802406,
      0.004590878263115883,
      0.011760087683796883,
      0.009128924459218979,
      0.17830823361873627,
      0.012547526508569717,
      0.02103581465780735,
      1.1236064665354206e-06
    ],
    "rationales": [
      "def",
      "_",
      "(",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " =",
      " \"",
      "\".",
      "join",
      "(",
      ".",
      "()"
    ],
    "rationales_indexes": [
      0,
      2,
      4,
      6,
      7,
      8,
      9,
      10,
      12,
      13,
      14,
      15,
      16,
      18,
      20
    ],
    "token": " for"
  },
  "22": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.011355943977832794,
      0.008700639009475708,
      0.0052477153949439526,
      0.00428914837539196,
      0.3808971047401428,
      0.05965051427483559,
      0.0011182163143530488,
      1.0416873919893987e-05
    ],
    "rationales": [
      "def",
      "pal",
      "s",
      " t",
      " =",
      " \"",
      "c",
      " for"
    ],
    "rationales_indexes": [
      0,
      3,
      5,
      11,
      12,
      13,
      17,
      21
    ],
    "token": " c"
  },
  "23": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.289159893989563,
      0.03203357756137848,
      0.05748414620757103,
      0.00291621801443398,
      2.238903834950179e-05
    ],
    "rationales": [
      "def",
      " =",
      ".",
      " for",
      " c"
    ],
    "rationales_indexes": [
      0,
      12,
      18,
      21,
      22
    ],
    "token": " in"
  },
  "24": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.003460364881902933,
      0.20616962015628815,
      0.011565588414669037,
      0.002161080250516534,
      0.006090535316616297,
      0.013028963468968868,
      0.013683061115443707,
      0.0136504415422678,
      0.016861185431480408,
      0.0019314425298944116,
      2.7191701519768685e-05
    ],
    "rationales": [
      "def",
      "(",
      "s",
      "join",
      "c",
      ".",
      "lower",
      "()",
      " for",
      " c",
      " in"
    ],
    "rationales_indexes": [
      0,
      4,
      5,
      15,
      17,
      18,
      19,
      20,
      21,
      22,
      23
    ],
    "token": " s"
  },
  "25": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.0020795671734958887,
      0.011853983625769615,
      0.05098259449005127,
      0.046537213027477264,
      0.04323478788137436,
      0.05529120936989784,
      0.004328196868300438,
      0.052432138472795486,
      0.04147384315729141,
      0.027944181114435196,
      0.05127869173884392,
      0.02361617051064968,
      0.05976423621177673,
      0.021667618304491043,
      0.03268100693821907,
      0.014904485084116459,
      0.0007582305697724223,
      0.05522645637392998,
      0.05208046734333038,
      0.04059367626905441,
      0.0525481291115284,
      0.01334447879344225,
      0.05065075308084488,
      0.022123614326119423,
      4.140678356634453e-05
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \"",
      "\".",
      "join",
      "(",
      "c",
      ".",
      "lower",
      "()",
      " for",
      " c",
      " in",
      " s"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24
    ],
    "token": " if"
  },
  "26": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.1963004469871521,
      0.4468996822834015,
      0.016580618917942047,
      3.192788062733598e-05
    ],
    "rationales": [
      "(",
      "c",
      " c",
      " if"
    ],
    "rationales_indexes": [
      4,
      17,
      22,
      25
    ],
    "token": " c"
  },
  "27": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Statements"
    ],
    "probabilities": [
      0.06051720678806305,
      0.23437947034835815,
      0.16881568729877472,
      0.1950906664133072,
      2.8282232960918918e-05
    ],
    "rationales": [
      "\".",
      "join",
      ".",
      "()",
      " c"
    ],
    "rationales_indexes": [
      14,
      15,
      18,
      20,
      26
    ],
    "token": "."
  },
  "28": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      5.01848489875556e-07,
      0.000326119625242427,
      0.0011252383701503277,
      0.00011584602179937065,
      0.0011707196244969964,
      0.0012902452144771814,
      0.0016535023460164666,
      0.0013578374637290835,
      0.0010949758579954505,
      0.0009144953219220042,
      0.0015414984663948417,
      0.0005842031096108258,
      0.0006753140478394926,
      9.114771160056989e-07,
      0.0001065749311237596,
      3.731371907633729e-05,
      0.0013508129632100463,
      0.0016297341790050268,
      0.0012327973963692784,
      1.0771091183414683e-05,
      0.0011492639314383268,
      0.00042507017496973276,
      0.0008108324836939573,
      0.0007938896305859089,
      0.0008001950918696821,
      3.2226576877292246e-05,
      9.692522144177929e-05,
      1.02365338428001e-10
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \"",
      "\".",
      "join",
      "(",
      "c",
      ".",
      "lower",
      "()",
      " for",
      " c",
      " in",
      " s",
      " if",
      " c",
      "."
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27
    ],
    "token": "isal"
  },
  "29": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.0001745532063068822,
      0.035155948251485825,
      0.02665039710700512,
      0.009744161739945412,
      0.00015589635586366057,
      0.0017667810898274183,
      0.04587380960583687,
      0.1784510463476181,
      0.075456403195858,
      3.654776321582176e-07
    ],
    "rationales": [
      "def",
      "s",
      " =",
      "c",
      "()",
      " for",
      " s",
      " if",
      ".",
      "isal"
    ],
    "rationales_indexes": [
      0,
      5,
      12,
      17,
      20,
      21,
      24,
      25,
      27,
      28
    ],
    "token": "num"
  },
  "3": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00014955326332710683,
      2.014699020946864e-05,
      1.025205165205989e-08
    ],
    "rationales": [
      "def",
      " is",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "pal"
  },
  "30": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.00012303722905926406,
      0.012011490762233734,
      0.05839398875832558,
      0.0018471392104402184,
      0.0002865653077606112,
      0.08780156075954437,
      0.11654819548130035,
      0.08284719288349152,
      0.0848783403635025,
      0.07725771516561508,
      0.06705233454704285,
      0.041802242398262024,
      0.03426545858383179,
      0.2339792400598526,
      0.0798882320523262,
      0.030048798769712448,
      0.013454764150083065,
      0.028195340186357498,
      0.018358858302235603,
      0.3175477683544159,
      0.21925672888755798,
      0.15556557476520538,
      0.1353870928287506,
      0.006571317091584206,
      0.04784726724028587,
      5.992623064443592e-12
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \"",
      "(",
      "c",
      ".",
      "lower",
      "()",
      " for",
      " c",
      " in",
      " s",
      ".",
      "isal",
      "num"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      27,
      28,
      29
    ],
    "token": "())"
  },
  "31": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.4010477364063263,
      0.0378204770386219
    ],
    "rationales": [
      "\".",
      "())"
    ],
    "rationales_indexes": [
      14,
      30
    ],
    "token": "\n"
  },
  "32": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.0001240202400367707,
      0.00021399765682872385,
      9.655247413320467e-05,
      0.001492413692176342,
      7.047885446809232e-05,
      0.00018865686433855444,
      0.00014363494119606912,
      0.0003758982347790152,
      9.124408097704872e-05,
      0.06717339158058167,
      0.03640815615653992,
      0.00019421886827331036,
      0.02207636646926403,
      0.0004489783605094999,
      0.0014472011243924499,
      0.10471352934837341,
      0.00037906100624240935,
      0.0006492227548733354,
      0.00037427752977237105,
      0.09186061471700668,
      0.08096209913492203,
      0.004938124679028988,
      0.0003685949486680329,
      0.0004867736715823412,
      0.0010207658633589745,
      0.0005487968446686864,
      0.0008533704094588757,
      0.000291152362478897,
      0.006133938208222389,
      0.001061849994584918,
      0.007027063053101301,
      3.775078880607907e-07
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \"",
      "\".",
      "join",
      "(",
      "c",
      ".",
      "lower",
      "()",
      " for",
      " c",
      " in",
      " s",
      " if",
      " c",
      ".",
      "isal",
      "num",
      "())",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31
    ],
    "token": " "
  },
  "33": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.9994258880615234
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      32
    ],
    "token": " "
  },
  "34": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9992619156837463
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      33
    ],
    "token": " "
  },
  "35": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00029348008683882654,
      0.035691998898983,
      0.1097051203250885,
      0.007169053424149752,
      0.014314933679997921,
      0.07090342044830322,
      0.001487902132794261,
      0.00021656366880051792,
      4.870839287463014e-10
    ],
    "rationales": [
      "def",
      "_",
      "pal",
      "):",
      " ",
      " t",
      " if",
      "())",
      " "
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      6,
      8,
      11,
      25,
      30,
      34
    ],
    "token": " return"
  },
  "36": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.05849241092801094,
      0.005505124107003212,
      0.035024501383304596,
      0.02234182320535183,
      0.34201228618621826,
      6.519157977891155e-07
    ],
    "rationales": [
      " t",
      "(",
      "c",
      " in",
      " ",
      " return"
    ],
    "rationales_indexes": [
      11,
      16,
      17,
      23,
      34,
      35
    ],
    "token": " t"
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.00020833515736740083,
      0.0014737186720594764,
      0.0029488930013030767,
      0.10467276722192764,
      0.013393080793321133,
      0.09403745830059052,
      0.041250474750995636,
      0.019271917641162872,
      0.12737055122852325,
      0.05357968062162399,
      0.2000204622745514,
      0.11887826025485992,
      0.0928427055478096,
      0.005208258051425219,
      0.12794911861419678,
      0.11148723214864731,
      0.13144731521606445,
      0.07242672145366669,
      4.184859056510781e-11
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "s",
      "):",
      " ",
      " t",
      " =",
      "(",
      "c",
      ".",
      "lower",
      "()",
      " if",
      " c",
      ".",
      "isal",
      "num",
      " t"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      5,
      6,
      10,
      11,
      12,
      16,
      17,
      18,
      19,
      20,
      25,
      26,
      27,
      28,
      29,
      36
    ],
    "token": " =="
  },
  "38": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.5819116234779358,
      2.3117027012631297e-06
    ],
    "rationales": [
      " t",
      " =="
    ],
    "rationales_indexes": [
      36,
      37
    ],
    "token": " t"
  },
  "39": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.0014353811275213957,
      0.13220348954200745,
      0.007442964240908623,
      0.1202145665884018,
      0.13264776766300201,
      0.01976863667368889,
      0.0062468396499753,
      0.1082606092095375,
      0.11542334407567978,
      0.11186517030000687,
      0.08880080282688141,
      0.015728561207652092,
      0.008776204660534859,
      0.03604534640908241,
      0.08086435496807098,
      0.08766689896583557,
      0.06830406188964844,
      0.1324453353881836,
      0.12822125852108002,
      0.09712515771389008,
      0.06027667224407196,
      0.05404910817742348,
      0.10471297055482864,
      0.046710897237062454,
      0.10813092440366745,
      0.02462081052362919,
      0.1120041012763977,
      0.033263497054576874,
      0.11295714974403381,
      0.10030513256788254,
      0.06645707041025162,
      0.09452307969331741,
      0.04009798541665077,
      0.04738626629114151,
      0.061880286782979965,
      0.03445560485124588,
      0.05939340591430664,
      0.05787743628025055,
      7.1526962130974425e-09
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \"",
      "\".",
      "join",
      "(",
      "c",
      ".",
      "lower",
      "()",
      " for",
      " c",
      " in",
      " s",
      " if",
      " c",
      ".",
      "isal",
      "num",
      "())",
      "\n",
      " ",
      " ",
      " ",
      " return",
      " t",
      " ==",
      " t"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38
    ],
    "token": "["
  },
  "4": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0037524583749473095,
      0.005611285101622343,
      0.004838863387703896,
      1.0773188981530457e-12
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "("
  },
  "40": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.00022737440303899348,
      0.006088452413678169,
      0.012610746547579765,
      0.003468069713562727,
      0.008628655225038528,
      0.012461996637284756,
      0.008827447891235352,
      0.003748377086594701,
      0.013104429468512535,
      0.013166680932044983,
      0.011940364725887775,
      0.007871054112911224,
      0.00206934567540884,
      0.007380189839750528,
      0.005104178097099066,
      0.0018454189412295818,
      0.016726180911064148,
      0.011315908282995224,
      0.007224626373499632,
      0.01468195952475071,
      0.011019837111234665,
      0.012568491511046886,
      0.01606696844100952,
      0.017883995547890663,
      0.009118788875639439,
      0.008079765364527702,
      0.009758670814335346,
      0.007554017473012209,
      0.006830853875726461,
      0.010764682665467262,
      0.015235159546136856,
      0.009926497004926205,
      0.021162768825888634,
      0.020132839679718018,
      0.018435625359416008,
      0.0196182020008564,
      0.0008725235238671303,
      0.013003651052713394,
      0.0008400131482630968,
      3.933732841687743e-06
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \"",
      "\".",
      "join",
      "(",
      "c",
      ".",
      "lower",
      "()",
      " for",
      " c",
      " in",
      " s",
      " if",
      " c",
      ".",
      "isal",
      "num",
      "())",
      "\n",
      " ",
      " ",
      " ",
      " return",
      " t",
      " ==",
      " t",
      "["
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39
    ],
    "token": "::"
  },
  "41": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.036750391125679016,
      0.03941820189356804,
      0.04914460331201553,
      0.07906736433506012,
      0.031996842473745346,
      0.10945693403482437,
      0.0438615158200264,
      0.05860904976725578,
      0.11260553449392319,
      0.033387843519449234,
      0.0345083586871624,
      0.04766135290265083,
      0.051376353949308395,
      0.04078172892332077,
      0.0366828516125679,
      0.03275791555643082,
      0.0490860641002655,
      0.11456365138292313,
      0.04716844484210014,
      0.06284371763467789,
      0.034576982259750366,
      0.057348936796188354,
      0.04596821218729019,
      0.09241975843906403,
      0.03998623043298721,
      0.10555807501077652,
      0.11733993887901306,
      0.10831161588430405,
      0.09906187653541565,
      0.05599753186106682,
      0.06421458721160889,
      2.9433401778078405e-06
    ],
    "rationales": [
      "_",
      "pal",
      "(",
      "s",
      " ",
      " ",
      " ",
      " t",
      " =",
      " \"",
      "\".",
      "join",
      "(",
      "c",
      ".",
      "lower",
      " for",
      " c",
      " in",
      " s",
      " if",
      " c",
      ".",
      "isal",
      "\n",
      " ",
      " ",
      " ",
      " return",
      " t",
      " ==",
      "::"
    ],
    "rationales_indexes": [
      2,
      3,
      4,
      5,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      40
    ],
    "token": "-"
  },
  "42": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.2111852467060089,
      0.22335213422775269,
      0.26648035645484924,
      0.0072767473757267,
      0.03008781559765339,
      0.18945345282554626,
      0.06445729732513428,
      4.804853233508766e-05
    ],
    "rationales": [
      "(",
      "s",
      "join",
      "(",
      "lower",
      " in",
      " if",
      "-"
    ],
    "rationales_indexes": [
      4,
      5,
      15,
      16,
      19,
      23,
      25,
      41
    ],
    "token": "1"
  },
  "43": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      0.5607707500457764,
      0.35647591948509216,
      0.1782773733139038,
      0.00013245921581983566
    ],
    "rationales": [
      " c",
      "())",
      "[",
      "1"
    ],
    "rationales_indexes": [
      26,
      30,
      39,
      42
    ],
    "token": "]"
  },
  "44": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.32712557911872864,
      0.0021094607654958963
    ],
    "rationales": [
      "def",
      "]"
    ],
    "rationales_indexes": [
      0,
      43
    ],
    "token": "\n"
  },
  "45": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9962778687477112
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      44
    ],
    "token": "\n"
  },
  "46": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.0006424618768505752,
      0.07419978827238083,
      0.013190983794629574,
      0.17140649259090424,
      1.5490061455736281e-09
    ],
    "rationales": [
      "def",
      "_",
      "):",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      0,
      2,
      6,
      35,
      45
    ],
    "token": "def"
  },
  "47": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.00028176841442473233,
      0.002315848832949996,
      0.004038106184452772,
      0.0032368528191000223,
      0.002907649613916874,
      0.0010750904912129045,
      0.0036262294743210077,
      0.0015933467075228691,
      0.0006572892889380455,
      0.01756995916366577,
      0.0020520519465208054,
      0.0330108106136322,
      0.007373802363872528,
      0.0042402553372085094,
      1.7061000789908576e-07
    ],
    "rationales": [
      "def",
      "(",
      "\n",
      " =",
      "join",
      "lower",
      " for",
      " in",
      " if",
      "())",
      "\n",
      " return",
      "]",
      "\n",
      "def"
    ],
    "rationales_indexes": [
      0,
      4,
      7,
      12,
      15,
      19,
      21,
      23,
      25,
      30,
      31,
      35,
      43,
      44,
      46
    ],
    "token": " get"
  },
  "48": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.008267011493444443,
      0.18724970519542694,
      0.08944851905107498,
      0.004887981805950403,
      1.2445105994629557e-06
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "::",
      " get"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      40,
      47
    ],
    "token": "_"
  },
  "49": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.001190869603306055,
      0.00025334637030027807,
      0.13802184164524078,
      0.00011617936979746446,
      0.2157006710767746,
      0.055469248443841934,
      8.284532668767497e-05,
      0.006519601214677095,
      2.7836367877398516e-08
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      " ",
      " =",
      "num",
      "def",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      8,
      12,
      29,
      46,
      48
    ],
    "token": "pal"
  },
  "5": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.003419723128899932,
      0.009565032087266445,
      0.007006244268268347,
      0.00981092732399702,
      8.780310599831864e-05
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "s"
  },
  "50": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.025086721405386925,
      0.13667906820774078,
      0.07909140735864639,
      0.10761718451976776,
      0.004121965728700161,
      0.0128202298656106,
      0.013770444318652153,
      0.011571557261049747,
      0.010271389968693256,
      1.6070973761841967e-12
    ],
    "rationales": [
      "def",
      "num",
      "())",
      " return",
      " ==",
      " t",
      "[",
      "\n",
      "def",
      "pal"
    ],
    "rationales_indexes": [
      0,
      29,
      30,
      35,
      37,
      38,
      39,
      44,
      46,
      49
    ],
    "token": "("
  },
  "51": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.05730122700333595,
      0.03305564448237419,
      0.025543933734297752,
      0.02465461567044258,
      0.2163095325231552,
      0.033287305384874344,
      0.0415828563272953,
      0.04520018398761749,
      0.05528922751545906,
      0.052455686032772064,
      0.06468603014945984,
      0.017949866130948067,
      0.019202256575226784,
      0.0023583287838846445,
      0.035569287836551666,
      0.029680751264095306,
      7.852541602915153e-05
    ],
    "rationales": [
      "(",
      " s",
      "\n",
      " ==",
      " t",
      "[",
      "::",
      "-",
      "1",
      "]",
      "\n",
      "\n",
      "def",
      " get",
      "_",
      "pal",
      "("
    ],
    "rationales_indexes": [
      16,
      24,
      31,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50
    ],
    "token": "s"
  },
  "52": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.0015033447416499257,
      0.08738115429878235,
      0.0031473631970584393,
      0.04848945140838623,
      0.16726842522621155,
      8.78542323334841e-06
    ],
    "rationales": [
      "def",
      "pal",
      "):",
      "(",
      "isal",
      "s"
    ],
    "rationales_indexes": [
      0,
      3,
      6,
      16,
      28,
      51
    ],
    "token": "):"
  },
  "53": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.9098907113075256,
      0.14990592002868652
    ],
    "rationales": [
      " \"",
      "):"
    ],
    "rationales_indexes": [
      13,
      52
    ],
    "token": "\n"
  },
  "54": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.9977667331695557
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      53
    ],
    "token": "\n"
  },
  "6": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.0023190281353890896,
      0.045353300869464874,
      0.052317604422569275,
      0.04295429214835167,
      0.014393526129424572,
      2.7063149943273856e-08
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "token": "):"
  },
  "7": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.8355810642242432,
      0.06029180809855461
    ],
    "rationales": [
      "s",
      "):"
    ],
    "rationales_indexes": [
      5,
      6
    ],
    "token": "\n"
  },
  "8": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      1.9623598745965865e-06,
      1.8470349232302397e-06,
      2.584804860816803e-06,
      4.397465454530902e-05,
      4.3014697439502925e-05,
      2.9778724638163112e-05,
      2.3235572371049784e-05,
      1.7774368643586058e-08
    ],
    "rationales": [
      "def",
      " is",
      "_",
      "pal",
      "(",
      "s",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ],
    "token": " "
  },
  "9": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.999458372592926
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      8
    ],
    "token": " "
  },
  "_phrase": "def is_pal(s):\n    t = \"\".join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n\ndef get_pal(s):\n\n"
}
curl -X POST http://127.0.0.1:5000/prompt -H "Content-Type: application/json"  0.02s user 0.02s system 0% cpu 1:10.21 total



Sample 8

time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def invert_map(m):\n    out = {}\n    for k, v in m.items():\n        out[v] = k\n    return out\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.004643934778869152
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " in"
  },
  "10": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9994294047355652
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      9
    ],
    "token": " "
  },
  "11": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.999504566192627
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      10
    ],
    "token": " "
  },
  "12": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "OOP"
    ],
    "probabilities": [
      3.208129055565223e-05,
      9.12902396521531e-05,
      0.00015521918248850852,
      2.3639593564439565e-05,
      0.0002165566838812083,
      2.131035944330506e-05,
      0.00019278796389698982,
      0.00022496013843920082,
      0.0011316909221932292,
      0.0004207781166769564,
      8.40076245367527e-05,
      1.745732980396042e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11
    ],
    "token": " out"
  },
  "13": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      7.628959428984672e-05,
      0.08125164359807968,
      0.20127242803573608,
      3.882590959847221e-08
    ],
    "rationales": [
      "def",
      "):",
      "\n",
      " out"
    ],
    "rationales_indexes": [
      0,
      7,
      8,
      12
    ],
    "token": " ="
  },
  "14": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Loops"
    ],
    "probabilities": [
      0.0008082770509645343,
      0.0023207226768136024,
      0.0024328669533133507,
      0.005156738217920065,
      0.004494716878980398,
      0.0053999461233615875,
      0.003760020947083831,
      0.003360687755048275,
      0.004858205560594797,
      0.004657674580812454,
      0.004899024963378906,
      0.0046469480730593204,
      0.004808308091014624,
      1.5666330455132993e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " ="
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13
    ],
    "token": " {}"
  },
  "15": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.30411338806152344,
      5.6970959121827036e-05
    ],
    "rationales": [
      "):",
      " {}"
    ],
    "rationales_indexes": [
      7,
      14
    ],
    "token": "\n"
  },
  "16": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.002940824721008539,
      0.3361113965511322,
      3.132145209860937e-08
    ],
    "rationales": [
      " ",
      " =",
      "\n"
    ],
    "rationales_indexes": [
      11,
      13,
      15
    ],
    "token": " "
  },
  "17": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "OOP"
    ],
    "probabilities": [
      0.9994862079620361
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      16
    ],
    "token": " "
  },
  "18": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.9994465708732605
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      17
    ],
    "token": " "
  },
  "19": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.000744360382668674,
      0.0945507064461708,
      0.0015791154000908136,
      0.1040189117193222,
      0.11701187491416931,
      0.00030885133310221136,
      0.0020062667317688465,
      0.04276895523071289,
      0.0011932294582948089,
      0.0008443064871244133,
      0.06683024019002914,
      0.002113957656547427,
      0.004409159068018198,
      0.0028162517119199038,
      0.011830441653728485,
      0.0025281235575675964,
      0.002341565676033497,
      0.0031922217458486557,
      1.4283772031831177e-07
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18
    ],
    "token": " for"
  },
  "2": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0001328982471022755,
      2.1865679839666585e-12
    ],
    "rationales": [
      "def",
      " in"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "vert"
  },
  "20": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Bool"
    ],
    "probabilities": [
      0.0005359220667742193,
      0.06874440610408783,
      0.04714268445968628,
      0.048507824540138245,
      0.048284128308296204,
      0.06750740110874176,
      0.036193713545799255,
      0.060286328196525574,
      0.06754721701145172,
      0.06639983505010605,
      0.06150505319237709,
      0.05147179588675499,
      0.02545998990535736,
      0.008353379555046558,
      0.03358158841729164,
      0.04551726579666138,
      0.05807099863886833,
      0.05443081632256508,
      0.055659856647253036,
      3.6085125429963227e-06
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " ",
      " for"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19
    ],
    "token": " k"
  },
  "21": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Signature"
    ],
    "probabilities": [
      0.05337857827544212,
      0.06544661521911621,
      0.05365435406565666,
      0.06318067759275436,
      0.0714430958032608,
      0.48930981755256653,
      0.284047394990921,
      0.1066080704331398,
      1.5387853636639193e-05
    ],
    "rationales": [
      "def",
      "vert",
      "_",
      "):",
      " out",
      " =",
      " ",
      " for",
      " k"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      7,
      12,
      13,
      18,
      19,
      20
    ],
    "token": ","
  },
  "22": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00023455919290427119,
      0.08421050757169724,
      0.03233680874109268,
      0.00718175433576107,
      0.002347260946407914,
      0.0014870527666062117,
      4.814361545868451e-06
    ],
    "rationales": [
      "def",
      "):",
      " =",
      " ",
      " for",
      " k",
      ","
    ],
    "rationales_indexes": [
      0,
      7,
      13,
      18,
      19,
      20,
      21
    ],
    "token": " v"
  },
  "23": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.13828237354755402,
      0.025381973013281822,
      0.0175801832228899,
      0.014389775693416595,
      0.007423491217195988,
      6.403104180208175e-07
    ],
    "rationales": [
      "def",
      "(",
      "):",
      " {}",
      " for",
      " v"
    ],
    "rationales_indexes": [
      0,
      5,
      7,
      14,
      19,
      22
    ],
    "token": " in"
  },
  "24": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.0009837507968768477,
      0.0712483748793602,
      0.004469496197998524,
      0.12820665538311005,
      0.2031295746564865,
      6.511116225738078e-05
    ],
    "rationales": [
      "def",
      " in",
      "m",
      " ",
      " k",
      " in"
    ],
    "rationales_indexes": [
      0,
      1,
      6,
      10,
      20,
      23
    ],
    "token": " m"
  },
  "25": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.04216836392879486,
      0.16170543432235718,
      0.03932971879839897,
      0.0549863800406456,
      0.04547853395342827,
      0.02970646694302559,
      0.0358988493680954,
      0.030591530725359917,
      0.04418950900435448,
      1.7822168274506112e-07
    ],
    "rationales": [
      "def",
      "_",
      "m",
      " out",
      " for",
      " k",
      ",",
      " v",
      " in",
      " m"
    ],
    "rationales_indexes": [
      0,
      3,
      6,
      12,
      19,
      20,
      21,
      22,
      23,
      24
    ],
    "token": "."
  },
  "26": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      1.0577382454357576e-05,
      0.0002999488788191229,
      0.01779025048017502,
      0.03174738585948944,
      0.0001111763485823758,
      0.007302064914256334,
      0.0008795924950391054,
      1.3446843638575956e-08
    ],
    "rationales": [
      "def",
      "_",
      "map",
      "(",
      "):",
      " for",
      " in",
      "."
    ],
    "rationales_indexes": [
      0,
      3,
      4,
      5,
      7,
      19,
      23,
      25
    ],
    "token": "items"
  },
  "27": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.00012925370538141578,
      0.44555848836898804,
      0.00248103984631598,
      0.0009109788225032389,
      0.06143005192279816,
      0.005045786499977112,
      0.013931289315223694,
      7.60669394139768e-09
    ],
    "rationales": [
      "def",
      "_",
      "m",
      "):",
      " for",
      " in",
      ".",
      "items"
    ],
    "rationales_indexes": [
      0,
      3,
      6,
      7,
      19,
      23,
      25,
      26
    ],
    "token": "():"
  },
  "28": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.7252012491226196,
      5.607935236184858e-05
    ],
    "rationales": [
      "\n",
      "():"
    ],
    "rationales_indexes": [
      15,
      27
    ],
    "token": "\n"
  },
  "29": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.4665857255458832,
      0.6970933079719543,
      0.1275458037853241,
      0.01514727994799614,
      0.00015423486183863133,
      2.8258369866307476e-07
    ],
    "rationales": [
      " ",
      " =",
      " ",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      10,
      13,
      16,
      17,
      18,
      28
    ],
    "token": " "
  },
  "3": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.049718860536813736,
      0.08333893865346909,
      1.4042745632991682e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "_"
  },
  "30": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9993077516555786
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      29
    ],
    "token": " "
  },
  "31": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.9993191957473755
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      30
    ],
    "token": " "
  },
  "32": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9992952346801758
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      31
    ],
    "token": " "
  },
  "33": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9994258880615234
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      32
    ],
    "token": " "
  },
  "34": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Functional"
    ],
    "probabilities": [
      0.9992619156837463
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      33
    ],
    "token": " "
  },
  "35": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9993482232093811
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      34
    ],
    "token": " "
  },
  "36": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "OOP"
    ],
    "probabilities": [
      1.2821828022424597e-05,
      0.012008055113255978,
      0.030686737969517708,
      0.032044824212789536,
      0.006452462170273066,
      0.0319024957716465,
      0.017542436718940735,
      0.007171932607889175,
      0.022599073126912117,
      0.007596177514642477,
      0.0046691601164639,
      0.0026513852644711733,
      5.836285345139913e-05,
      0.016830191016197205,
      3.0041335776331834e-05,
      0.019553575664758682,
      0.01924770697951317,
      0.02092747576534748,
      0.032037340104579926,
      0.020259113982319832,
      0.029505455866456032,
      2.1611203919746913e-05,
      4.4300548324827105e-05,
      0.0009630157728679478,
      2.5102122890530154e-05,
      0.0029600574634969234,
      0.011658593080937862,
      3.9564551116200164e-05,
      0.001475475961342454,
      0.0003140124026685953,
      0.0001053667874657549,
      0.0007547980058006942,
      1.9650613467092626e-05,
      0.0013440856710076332,
      0.0018687908304855227,
      4.02741449079258e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " k",
      ",",
      " v",
      " in",
      " m",
      ".",
      "items",
      "():",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35
    ],
    "token": " out"
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00013292237417772412,
      0.011900685727596283,
      0.232132226228714,
      0.007591916248202324,
      0.023323697969317436,
      0.12098576128482819,
      0.1529783308506012,
      0.0028360954020172358,
      0.08909974247217178,
      0.16069310903549194,
      0.1526115983724594,
      0.1974925845861435,
      0.03629600256681442,
      0.01835571974515915,
      0.06356427818536758,
      0.25299206376075745,
      0.17580334842205048,
      0.182384192943573,
      0.1912044882774353,
      0.25533849000930786,
      0.17417801916599274,
      0.16635474562644958,
      0.260482519865036,
      0.2524064779281616,
      0.24851222336292267,
      0.07848816365003586,
      0.24035505950450897,
      0.2148548662662506,
      0.21895462274551392,
      0.22711512446403503,
      0.22500158846378326,
      0.22111892700195312,
      0.2116222232580185,
      0.18008682131767273,
      0.12983043491840363,
      0.10345269739627838,
      6.911163268341625e-07
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " k",
      ",",
      " v",
      " in",
      " m",
      ".",
      "items",
      "():",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " out"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36
    ],
    "token": "["
  },
  "38": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.0015499723376706243,
      0.046976979821920395,
      0.018821682780981064,
      0.07352974265813828,
      0.0307348370552063,
      0.010044409893453121,
      5.448785486805718e-06
    ],
    "rationales": [
      "def",
      "m",
      "):",
      " k",
      ",",
      " v",
      "["
    ],
    "rationales_indexes": [
      0,
      6,
      7,
      20,
      21,
      22,
      37
    ],
    "token": "v"
  },
  "39": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.539456844329834,
      0.16760021448135376,
      1.6001536096155178e-07
    ],
    "rationales": [
      " out",
      "[",
      "v"
    ],
    "rationales_indexes": [
      36,
      37,
      38
    ],
    "token": "]"
  },
  "4": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.0019030083203688264,
      0.0029502855613827705,
      0.0018593764398247004,
      9.793885169528949e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "map"
  },
  "40": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.0036984614562243223,
      0.03566648066043854,
      0.24974505603313446,
      0.00014856824418529868
    ],
    "rationales": [
      "def",
      " =",
      "[",
      "]"
    ],
    "rationales_indexes": [
      0,
      13,
      37,
      39
    ],
    "token": " ="
  },
  "41": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.0005403705290518701,
      0.038557518273591995,
      0.04668039828538895,
      0.015278986655175686,
      0.10347408801317215,
      0.02807946503162384,
      0.02191704884171486,
      0.040933385491371155,
      0.06478850543498993,
      0.03481868654489517,
      0.08837969601154327,
      0.008982685394585133,
      0.05409364402294159,
      1.9055884195040562e-06
    ],
    "rationales": [
      "def",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " {}",
      " ",
      " k",
      ".",
      " ="
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      11,
      14,
      18,
      20,
      25,
      40
    ],
    "token": " k"
  },
  "42": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.23339831829071045,
      7.0017749749240465e-06
    ],
    "rationales": [
      " =",
      " k"
    ],
    "rationales_indexes": [
      40,
      41
    ],
    "token": "\n"
  },
  "43": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.11968293786048889,
      0.9676706194877625,
      0.0031396362464874983,
      1.7619690595438442e-07
    ],
    "rationales": [
      "():",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      27,
      30,
      35,
      42
    ],
    "token": " "
  },
  "44": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "OOP"
    ],
    "probabilities": [
      0.9993201494216919
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      43
    ],
    "token": " "
  },
  "45": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9993821382522583
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      44
    ],
    "token": " "
  },
  "46": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.0005735348677262664,
      0.06359754502773285,
      0.00023117601813282818,
      0.04623054713010788,
      0.0006697351927869022,
      0.04288335517048836,
      0.00014329208352137357,
      0.03631835803389549,
      0.0005031678010709584,
      0.05738892778754234,
      0.050135254859924316,
      0.0007319389842450619,
      0.0006277281208895147,
      0.0006553889834322035,
      0.039181191474199295,
      0.06721752136945724,
      0.0006288556614890695,
      0.032827261835336685,
      0.00018605927471071482,
      0.026025619357824326,
      0.0647859126329422,
      0.0005777011974714696,
      0.07378644496202469,
      0.00010942455264739692,
      0.07471545040607452,
      0.06770730763673782,
      0.0006434098468162119,
      7.709614146733657e-05,
      0.010117716155946255,
      0.0007172901532612741,
      0.0007170590688474476,
      0.0006573807331733406,
      0.0005393070168793201,
      0.0003819090488832444,
      0.00013108378334436566,
      5.082256029709242e-05,
      0.05850696936249733,
      4.1043102100957185e-05,
      0.00017781245696824044,
      0.012311731465160847,
      0.0002009596355492249,
      0.00011911686306120828,
      0.0029179707635194063,
      8.909777534427121e-05,
      0.07351817935705185,
      2.1768023494850297e-10
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " k",
      ",",
      " v",
      " in",
      " m",
      ".",
      "items",
      "():",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " out",
      "[",
      "v",
      "]",
      " =",
      " k",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45
    ],
    "token": " return"
  },
  "47": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.000828124932013452,
      0.11929982155561447,
      6.19405818724772e-06
    ],
    "rationales": [
      " out",
      "\n",
      " return"
    ],
    "rationales_indexes": [
      36,
      42,
      46
    ],
    "token": " out"
  },
  "48": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.3323479890823364,
      1.4756413293071091e-05
    ],
    "rationales": [
      "\n",
      " out"
    ],
    "rationales_indexes": [
      42,
      47
    ],
    "token": "\n"
  },
  "49": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.9968153834342957
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      48
    ],
    "token": "\n"
  },
  "5": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.09929046779870987,
      0.06787649542093277,
      0.06354615837335587,
      0.07740329951047897,
      1.2816850869512564e-07
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "("
  },
  "50": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.0005895178182981908,
      0.18169361352920532,
      0.009976380504667759,
      0.06603942811489105,
      1.247240422053153e-09
    ],
    "rationales": [
      "def",
      "_",
      "):",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      0,
      3,
      7,
      46,
      49
    ],
    "token": "def"
  },
  "51": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.00026455268380232155,
      0.0005871982430107892,
      0.0034008584916591644,
      0.0038843797519803047,
      0.00103426119312644,
      0.0016013940330594778,
      0.011739395558834076,
      0.05466322600841522,
      0.003608484286814928,
      0.051090944558382034,
      0.04534303769469261,
      0.04100764915347099,
      0.0025816974230110645,
      0.03167170658707619,
      0.007365423254668713,
      0.020363513380289078,
      0.004228613339364529,
      1.6607063457740878e-07
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      " out",
      " =",
      " {}",
      " for",
      ".",
      "items",
      "():",
      " out",
      "]",
      " return",
      " out",
      "\n",
      "def"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      12,
      13,
      14,
      19,
      25,
      26,
      27,
      36,
      39,
      46,
      47,
      49,
      50
    ],
    "token": " get"
  },
  "52": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.004263021983206272,
      0.15532748401165009,
      0.07462489604949951,
      1.1989957329205936e-06
    ],
    "rationales": [
      "def",
      " in",
      "_",
      " get"
    ],
    "rationales_indexes": [
      0,
      1,
      3,
      51
    ],
    "token": "_"
  },
  "53": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.0004419742326717824,
      0.007392094470560551,
      3.0022811188246123e-06,
      0.0884210541844368,
      0.15155373513698578,
      0.03989620879292488,
      1.1383962217337285e-08
    ],
    "rationales": [
      "def",
      "_",
      "map",
      "):",
      "\n",
      " get",
      "_"
    ],
    "rationales_indexes": [
      0,
      3,
      4,
      7,
      28,
      51,
      52
    ],
    "token": "map"
  },
  "54": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.011109177023172379,
      0.06612485647201538,
      0.07043087482452393,
      0.05479004979133606,
      0.15867973864078522,
      0.09027814865112305,
      0.0712502971291542,
      1.5708549483406387e-07
    ],
    "rationales": [
      "def",
      "_",
      "m",
      "):",
      " out",
      " =",
      "\n",
      "map"
    ],
    "rationales_indexes": [
      0,
      3,
      6,
      7,
      12,
      13,
      15,
      53
    ],
    "token": "("
  },
  "6": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Loops"
    ],
    "probabilities": [
      0.002673191949725151,
      0.0073858448304235935,
      0.005867123603820801,
      0.010317721404135227,
      0.007397271692752838,
      9.167608368443325e-05
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "token": "m"
  },
  "7": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Loops"
    ],
    "probabilities": [
      0.0008127083419822156,
      0.0015545116038993,
      0.0033387800212949514,
      0.0036388200242072344,
      0.002361996565014124,
      0.0017948491731658578,
      5.450808207574376e-11
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6
    ],
    "token": "):"
  },
  "8": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.8472586870193481,
      0.06658641248941422
    ],
    "rationales": [
      "vert",
      "):"
    ],
    "rationales_indexes": [
      2,
      7
    ],
    "token": "\n"
  },
  "9": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      1.835906004998833e-06,
      1.4819322586845374e-06,
      7.218062819447368e-06,
      2.6093657652381808e-06,
      1.7579928680788726e-05,
      5.1595852710306644e-05,
      1.249426986760227e-05,
      1.0562840543570928e-05,
      1.7788709172350536e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    "token": " "
  },
  "_phrase": "def invert_map(m):\n    out = {}\n    for k, v in m.items():\n        out[v] = k\n    return out\n\ndef get_map("
}
curl -X POST http://127.0.0.1:5000/prompt -H "Content-Type: application/json"  0.02s user 0.02s system 0% cpu 49.388 total


Sample 9

time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def take_while_pos(nums):\n    res = []\n    for n in nums:\n        if n <= 0:\n            break\n        res.append(n)\n    return res\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.00022739423729944974
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " take"
  },
  "10": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.18303488194942474,
      0.06292856484651566
    ],
    "rationales": [
      "def",
      "):"
    ],
    "rationales_indexes": [
      0,
      9
    ],
    "token": "\n"
  },
  "11": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      6.165915351630247e-07,
      2.2730453110852977e-06,
      1.5877421901677735e-05,
      1.3950991160527337e-05,
      3.078920053667389e-05,
      3.324718818475958e-06,
      6.895029946463183e-05,
      9.514447810943238e-06,
      1.8285432815901004e-05,
      2.5007343538163695e-06,
      1.6497244459401372e-08
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_",
      "pos",
      "(",
      "n",
      "ums",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10
    ],
    "token": " "
  },
  "12": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9994699358940125
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      11
    ],
    "token": " "
  },
  "13": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.9994625449180603
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      12
    ],
    "token": " "
  },
  "14": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      7.884917977207806e-06,
      0.0015985396457836032,
      0.0021473264787346125,
      2.3154370865086094e-05,
      0.000777064764406532,
      0.001995809143409133,
      0.0023439298383891582,
      0.0005832014721818268,
      0.0010829685488715768,
      0.00013054262672085315,
      0.0010274015367031097,
      5.333798981155269e-05,
      0.0013973077293485403,
      5.772485896400614e-13
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_",
      "pos",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13
    ],
    "token": " res"
  },
  "15": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.16174544394016266,
      6.434233029185832e-14
    ],
    "rationales": [
      "def",
      " res"
    ],
    "rationales_indexes": [
      0,
      14
    ],
    "token": " ="
  },
  "16": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0020695410203188658,
      0.03869866579771042,
      0.025764092803001404,
      0.030636539682745934,
      0.03836684674024582,
      0.01949056051671505,
      0.034431830048561096,
      0.0319814458489418,
      0.015154684893786907,
      0.009101763367652893,
      0.024953484535217285,
      0.02127147652208805,
      0.034628111869096756,
      0.037132371217012405,
      0.0046627032570540905,
      1.1219826490105334e-07
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_",
      "pos",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " res",
      " ="
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15
    ],
    "token": " []"
  },
  "17": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.42811793088912964,
      3.299518880339747e-08
    ],
    "rationales": [
      "\n",
      " []"
    ],
    "rationales_indexes": [
      10,
      16
    ],
    "token": "\n"
  },
  "18": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.38536548614501953,
      0.0018563197227194905,
      0.3084595799446106,
      6.947907849053081e-08
    ],
    "rationales": [
      " ",
      " ",
      " =",
      "\n"
    ],
    "rationales_indexes": [
      11,
      13,
      15,
      17
    ],
    "token": " "
  },
  "19": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.9994450211524963
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      18
    ],
    "token": " "
  },
  "2": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.005436540115624666,
      2.2765584990064314e-12
    ],
    "rationales": [
      "def",
      " take"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "_"
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.999420166015625
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      19
    ],
    "token": " "
  },
  "21": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.00045928568579256535,
      0.25822752714157104,
      0.0003313036577310413,
      0.007479763589799404,
      0.00639759749174118,
      0.09672833234071732,
      0.003976787440478802,
      0.004039682447910309,
      0.001058149035088718,
      0.18306201696395874,
      0.007809621747583151,
      0.00194805976934731,
      0.003473510965704918,
      0.002848420524969697,
      0.0029065420385450125,
      0.004979223478585482,
      1.771273048234434e-07
    ],
    "rationales": [
      "def",
      "while",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " res",
      " =",
      " []",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      3,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20
    ],
    "token": " for"
  },
  "22": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.0010544672841206193,
      0.02690109796822071,
      0.0033493221271783113,
      0.10420572012662888,
      6.365327408275334e-06
    ],
    "rationales": [
      "def",
      "(",
      "n",
      "\n",
      " for"
    ],
    "rationales_indexes": [
      0,
      6,
      7,
      17,
      21
    ],
    "token": " n"
  },
  "23": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.042788080871105194,
      0.009711598977446556,
      0.496966153383255,
      0.011204536072909832,
      0.011146776378154755,
      0.022617081180214882,
      1.668332697590813e-05
    ],
    "rationales": [
      "def",
      "ums",
      "):",
      "\n",
      " =",
      " for",
      " n"
    ],
    "rationales_indexes": [
      0,
      8,
      9,
      10,
      15,
      21,
      22
    ],
    "token": " in"
  },
  "24": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.00017475600179750472,
      0.016226474195718765,
      0.042863477021455765,
      0.010481851175427437,
      0.03153792396187782,
      0.03037351742386818,
      0.004146208055317402,
      0.03678508847951889,
      0.001723086112178862,
      0.03581569716334343,
      0.034551557153463364,
      0.038760650902986526,
      0.03333943337202072,
      0.03466497361660004,
      0.0027184830978512764,
      0.0354035384953022,
      0.023967262357473373,
      0.03770360350608826,
      0.04288458824157715,
      0.04581446945667267,
      0.043846502900123596,
      0.0007367784273810685,
      7.243583240779117e-05,
      6.511904757644515e-07
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_",
      "pos",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " res",
      " =",
      " []",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " n",
      " in"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23
    ],
    "token": " num"
  },
  "25": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.030330950394272804,
      0.15423841774463654,
      0.06008446589112282,
      2.6223342941555927e-11
    ],
    "rationales": [
      "def",
      "(",
      "ums",
      " num"
    ],
    "rationales_indexes": [
      0,
      6,
      8,
      24
    ],
    "token": "s"
  },
  "26": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.05655428394675255,
      0.09713965654373169,
      0.172296941280365,
      0.029123185202479362,
      0.24530991911888123,
      0.02740740031003952,
      1.4845790246909019e-05
    ],
    "rationales": [
      "ums",
      "):",
      "\n",
      " =",
      " []",
      "\n",
      "s"
    ],
    "rationales_indexes": [
      8,
      9,
      10,
      15,
      16,
      17,
      25
    ],
    "token": ":"
  },
  "27": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.21424183249473572,
      0.49685508012771606,
      3.5348259643797064e-06
    ],
    "rationales": [
      " n",
      " in",
      ":"
    ],
    "rationales_indexes": [
      22,
      23,
      26
    ],
    "token": "\n"
  },
  "28": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.8969734907150269,
      0.0005893128691241145,
      0.04567301645874977,
      2.3086349187906308e-07
    ],
    "rationales": [
      " ",
      " []",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      12,
      16,
      18,
      27
    ],
    "token": " "
  },
  "29": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.9992671608924866
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      28
    ],
    "token": " "
  },
  "3": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      4.708164851763286e-05,
      0.001637528301216662,
      4.164751743473971e-08
    ],
    "rationales": [
      "def",
      " take",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "while"
  },
  "30": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9993077516555786
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      29
    ],
    "token": " "
  },
  "31": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Asserts"
    ],
    "probabilities": [
      0.9993191957473755
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      30
    ],
    "token": " "
  },
  "32": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.9992952346801758
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      31
    ],
    "token": " "
  },
  "33": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.9994258880615234
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      32
    ],
    "token": " "
  },
  "34": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.9992619156837463
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      33
    ],
    "token": " "
  },
  "35": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.04883645847439766,
      0.1429923176765442,
      0.012691950425505638,
      0.0005217493744567037,
      0.02290498837828636,
      0.010166612453758717,
      0.09153251349925995,
      0.009090352803468704,
      0.04240000247955322,
      0.009172594174742699,
      0.02587130479514599,
      0.014115987345576286,
      0.026676593348383904,
      0.048166655004024506,
      0.024558862671256065,
      0.20401112735271454,
      0.1733807623386383,
      0.01027522049844265,
      0.026370029896497726,
      0.014301841147243977,
      0.025543546304106712,
      0.022211046889424324,
      0.00638387817889452,
      0.01699553057551384,
      0.02428942173719406,
      1.0751833912081565e-07
    ],
    "rationales": [
      "def",
      "while",
      "pos",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " res",
      " =",
      " []",
      "\n",
      " ",
      " ",
      " for",
      " n",
      " in",
      " num",
      "s",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      3,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      27,
      32,
      33,
      34
    ],
    "token": " if"
  },
  "36": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.46219947934150696,
      0.0013282309519127011,
      7.732250378467143e-05
    ],
    "rationales": [
      " n",
      ":",
      " if"
    ],
    "rationales_indexes": [
      22,
      26,
      35
    ],
    "token": " n"
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      2.1772964828414842e-05,
      0.12180738896131516,
      0.15485751628875732,
      0.005004779901355505,
      0.10334061086177826,
      0.1644396334886551,
      0.018382027745246887,
      0.009268385358154774,
      0.017274800688028336,
      0.0015521160094067454,
      0.12435474246740341,
      0.054029855877161026,
      0.054151151329278946,
      0.07097524404525757,
      0.00046262468094937503,
      9.416799730388448e-05,
      0.09220581501722336,
      0.10018211603164673,
      0.06664086878299713,
      0.0612310990691185,
      0.05660683661699295,
      0.03657829761505127,
      0.14517775177955627,
      0.047560714185237885,
      0.027156807482242584,
      0.0888051763176918,
      0.06249869614839554,
      0.07659238576889038,
      0.07018718868494034,
      0.07350147515535355,
      0.06304895132780075,
      0.05969223380088806,
      0.058400463312864304,
      0.05783462151885033,
      0.05789509415626526,
      0.0002506251912564039,
      3.568677797716191e-09
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_",
      "pos",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " res",
      " =",
      " []",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " n",
      " in",
      " num",
      "s",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " if",
      " n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36
    ],
    "token": " <="
  },
  "38": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.07736966013908386,
      0.09497471898794174,
      5.270992005534936e-06
    ],
    "rationales": [
      "(",
      " for",
      " <="
    ],
    "rationales_indexes": [
      6,
      21,
      37
    ],
    "token": " 0"
  },
  "39": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.0027064874302595854,
      0.08283904194831848,
      0.060436349362134933,
      0.17708849906921387,
      0.09364070743322372,
      0.14339490234851837,
      0.1045331284403801,
      0.00022975551837589592
    ],
    "rationales": [
      "def",
      "pos",
      "):",
      " ",
      " []",
      " num",
      ":",
      " 0"
    ],
    "rationales_indexes": [
      0,
      5,
      9,
      13,
      16,
      24,
      26,
      38
    ],
    "token": ":"
  },
  "4": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Statements"
    ],
    "probabilities": [
      0.173413947224617,
      0.0460650734603405,
      0.11756429821252823,
      1.696233266557101e-05
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "_"
  },
  "40": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.25640103220939636,
      0.3550886809825897,
      0.28183984756469727,
      0.9318870902061462,
      0.21315373480319977,
      2.8197027859278023e-06
    ],
    "rationales": [
      " n",
      " in",
      ":",
      "\n",
      " n",
      ":"
    ],
    "rationales_indexes": [
      22,
      23,
      26,
      27,
      36,
      39
    ],
    "token": "\n"
  },
  "41": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.09189768135547638,
      0.36041006445884705,
      0.0030995977576822042,
      2.5881317355924693e-07
    ],
    "rationales": [
      " num",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      24,
      33,
      34,
      40
    ],
    "token": " "
  },
  "42": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9992117881774902
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      41
    ],
    "token": " "
  },
  "43": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.999313473701477
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      42
    ],
    "token": " "
  },
  "44": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9993201494216919
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      43
    ],
    "token": " "
  },
  "45": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9993821382522583
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      44
    ],
    "token": " "
  },
  "46": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.9994580149650574
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      45
    ],
    "token": " "
  },
  "47": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      0.9993767142295837
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      46
    ],
    "token": " "
  },
  "48": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.999299168586731
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      47
    ],
    "token": " "
  },
  "49": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.9995001554489136
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      48
    ],
    "token": " "
  },
  "5": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.0004068778071086854,
      0.0005297403549775481,
      0.0005281063495203853,
      0.00025680888211354613,
      8.865502536536951e-07
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "pos"
  },
  "50": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9993146657943726
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      49
    ],
    "token": " "
  },
  "51": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.999398946762085
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      50
    ],
    "token": " "
  },
  "52": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      8.58341809362173e-06,
      0.028341732919216156,
      0.026410652324557304,
      0.0005711334524676204,
      0.026161320507526398,
      0.013622285798192024,
      0.018127189949154854,
      0.0008644863264635205,
      7.941317744553089e-05,
      0.0001557718205731362,
      0.029420645907521248,
      0.027021802961826324,
      0.024150878190994263,
      0.02840675227344036,
      0.003989436663687229,
      0.02572065405547619,
      0.03119768016040325,
      0.025725247338414192,
      0.026505036279559135,
      0.004796512890607119,
      0.003092607483267784,
      0.0016697932733222842,
      0.03077276609838009,
      0.017710503190755844,
      0.026095617562532425,
      0.00659798551350832,
      0.0022988012060523033,
      0.027328776195645332,
      0.023615382611751556,
      0.02237817831337452,
      0.011375775560736656,
      0.005954929627478123,
      0.008112514391541481,
      0.028166012838482857,
      0.0013046897947788239,
      4.6176399337127805e-05,
      0.029535934329032898,
      0.028174620121717453,
      0.02183481492102146,
      0.011247889138758183,
      0.031778715550899506,
      0.02677532471716404,
      0.026351463049650192,
      0.0275946706533432,
      0.0021140335593372583,
      0.0006336547085084021,
      0.0011123099830001593,
      0.003248464548960328,
      0.005616815760731697,
      0.022081630304455757,
      0.009401238523423672,
      7.889750519574079e-10
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_",
      "pos",
      "(",
      "n",
      "ums",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " res",
      " =",
      " []",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " n",
      " in",
      " num",
      "s",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " if",
      " n",
      " <=",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51
    ],
    "token": " break"
  },
  "53": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.10354147851467133,
      0.3194088041782379,
      4.619938863470452e-06
    ],
    "rationales": [
      "def",
      "\n",
      " break"
    ],
    "rationales_indexes": [
      0,
      10,
      52
    ],
    "token": "\n"
  },
  "54": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.6716946363449097,
      0.1986595243215561,
      0.0953836590051651,
      6.543690744820196e-08
    ],
    "rationales": [
      " num",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      24,
      50,
      51,
      53
    ],
    "token": " "
  },
  "55": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.9993746876716614
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      54
    ],
    "token": " "
  },
  "56": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9994377493858337
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      55
    ],
    "token": " "
  },
  "57": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9994470477104187
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      56
    ],
    "token": " "
  },
  "58": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9994180202484131
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      57
    ],
    "token": " "
  },
  "59": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9993622899055481
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      58
    ],
    "token": " "
  },
  "6": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.07709002494812012,
      3.4005655180635586e-08
    ],
    "rationales": [
      "def",
      "pos"
    ],
    "rationales_indexes": [
      0,
      5
    ],
    "token": "("
  },
  "60": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.9994662404060364
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      59
    ],
    "token": " "
  },
  "61": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Functional"
    ],
    "probabilities": [
      8.778086339589208e-06,
      0.004751560278236866,
      0.10681076347827911,
      5.8058107242686674e-05,
      0.00024368465528823435,
      0.0009836227400228381,
      2.865631180699185e-13
    ],
    "rationales": [
      "def",
      "pos",
      " ",
      " res",
      " =",
      " for",
      " "
    ],
    "rationales_indexes": [
      0,
      5,
      13,
      14,
      15,
      21,
      60
    ],
    "token": " res"
  },
  "62": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.05067683756351471,
      0.05582723021507263,
      0.14156192541122437,
      0.12228228896856308,
      0.23456352949142456,
      0.16719000041484833,
      0.4136218726634979,
      3.120538202838752e-11
    ],
    "rationales": [
      "def",
      "while",
      "n",
      "):",
      " res",
      " =",
      " []",
      " res"
    ],
    "rationales_indexes": [
      0,
      3,
      7,
      9,
      14,
      15,
      16,
      61
    ],
    "token": "."
  },
  "63": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      1.3800113265460823e-06,
      0.032387543469667435,
      0.000318195583531633,
      0.05167630314826965,
      0.10620328038930893,
      2.940139438578626e-06,
      2.3402583337883698e-06,
      2.1577337975031696e-05,
      0.0031452830880880356,
      0.01462241169065237,
      5.592086017713882e-05,
      4.3966655316474146e-10
    ],
    "rationales": [
      "def",
      " take",
      "pos",
      "):",
      " []",
      " n",
      " n",
      " <=",
      " break",
      " ",
      " res",
      "."
    ],
    "rationales_indexes": [
      0,
      1,
      5,
      9,
      16,
      22,
      36,
      37,
      52,
      60,
      61,
      62
    ],
    "token": "append"
  },
  "64": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.0662894994020462,
      0.27426689863204956,
      0.21328912675380707,
      0.13704709708690643,
      1.7834985044373752e-07
    ],
    "rationales": [
      "def",
      "_",
      "pos",
      "):",
      "append"
    ],
    "rationales_indexes": [
      0,
      2,
      5,
      9,
      63
    ],
    "token": "("
  },
  "65": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0015427485341206193,
      0.016340365633368492,
      0.030735723674297333,
      0.007936961948871613,
      4.2752144508995116e-05
    ],
    "rationales": [
      "def",
      " n",
      ".",
      "append",
      "("
    ],
    "rationales_indexes": [
      0,
      22,
      62,
      63,
      64
    ],
    "token": "n"
  },
  "66": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.019259309396147728,
      0.22889861464500427,
      1.1480012744868873e-06
    ],
    "rationales": [
      "def",
      "(",
      "n"
    ],
    "rationales_indexes": [
      0,
      6,
      65
    ],
    "token": ")"
  },
  "67": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.666038990020752,
      0.04060966148972511
    ],
    "rationales": [
      " res",
      ")"
    ],
    "rationales_indexes": [
      61,
      66
    ],
    "token": "\n"
  },
  "68": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.6032986044883728,
      0.2066470831632614,
      0.09918585419654846,
      0.016927890479564667,
      1.5307147549492583e-08
    ],
    "rationales": [
      " num",
      " ",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      24,
      57,
      59,
      60,
      67
    ],
    "token": " "
  },
  "69": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9994613528251648
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      68
    ],
    "token": " "
  },
  "7": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.00928206741809845,
      0.014540758915245533,
      0.02024674601852894,
      0.0174237247556448,
      0.018340276554226875,
      0.01237686350941658,
      4.6347558964043856e-05
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_",
      "pos",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6
    ],
    "token": "n"
  },
  "70": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.9994688630104065
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      69
    ],
    "token": " "
  },
  "71": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Functional"
    ],
    "probabilities": [
      4.312591045163572e-05,
      0.011022163555026054,
      0.06319200247526169,
      0.12785065174102783,
      0.03160898759961128,
      0.01884251832962036,
      0.07668997347354889,
      0.0062308842316269875,
      0.0014178791316226125,
      0.04637736827135086,
      0.00041155650978907943,
      0.003300845855847001,
      0.08956648409366608,
      1.5397197405953023e-10
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "_",
      "pos",
      "):",
      " n",
      " num",
      " if",
      " ",
      " break",
      " res",
      ")",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      4,
      5,
      9,
      22,
      24,
      35,
      41,
      52,
      61,
      66,
      70
    ],
    "token": " return"
  },
  "72": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.0007063239463604987,
      0.02165103890001774,
      0.2151329070329666,
      6.7012920226261485e-06
    ],
    "rationales": [
      "def",
      " res",
      "(",
      " return"
    ],
    "rationales_indexes": [
      0,
      14,
      64,
      71
    ],
    "token": " res"
  },
  "73": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.6471723914146423,
      1.3434964252212467e-08
    ],
    "rationales": [
      "\n",
      " res"
    ],
    "rationales_indexes": [
      67,
      72
    ],
    "token": "\n"
  },
  "74": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "With"
    ],
    "probabilities": [
      0.9993541836738586
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      73
    ],
    "token": "\n"
  },
  "75": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      0.0004928832058794796,
      0.038442905992269516,
      0.0033874944783747196,
      0.10738551616668701,
      2.182430972053062e-11
    ],
    "rationales": [
      "def",
      "_",
      "):",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      0,
      2,
      9,
      71,
      74
    ],
    "token": "def"
  },
  "76": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      4.7484154492849484e-05,
      0.04060329869389534,
      0.5753465294837952,
      2.878256744054397e-08
    ],
    "rationales": [
      "def",
      " take",
      " res",
      "def"
    ],
    "rationales_indexes": [
      0,
      1,
      72,
      75
    ],
    "token": " take"
  },
  "77": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.04754441976547241,
      0.8278942704200745,
      0.023952819406986237,
      0.03472797945141792,
      0.031053248792886734,
      0.043277956545352936,
      0.053479451686143875,
      0.028646349906921387,
      0.061919789761304855,
      0.034473538398742676,
      0.04150298237800598,
      0.0387662872672081,
      0.03107691928744316,
      0.06489598006010056,
      0.039065372198820114,
      0.04858480766415596,
      0.0382845513522625,
      0.00034459601738490164,
      1.0550147422350165e-08
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "_",
      "(",
      "\n",
      " res",
      " =",
      " for",
      " in",
      "\n",
      " 0",
      ":",
      ".",
      ")",
      "\n",
      "\n",
      "\n",
      " take"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      4,
      6,
      10,
      14,
      15,
      21,
      23,
      27,
      38,
      39,
      62,
      66,
      67,
      73,
      74,
      76
    ],
    "token": "_"
  },
  "78": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      7.2738921517157e-06,
      0.010757300071418285,
      0.1554979532957077,
      0.0019781161099672318,
      7.932676737709699e-09
    ],
    "rationales": [
      "def",
      "ums",
      "):",
      " take",
      "_"
    ],
    "rationales_indexes": [
      0,
      8,
      9,
      76,
      77
    ],
    "token": "while"
  },
  "79": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      0.6831934452056885,
      1.7233720427611843e-05
    ],
    "rationales": [
      "_",
      "while"
    ],
    "rationales_indexes": [
      77,
      78
    ],
    "token": "_"
  },
  "8": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      8.204876212403178e-05,
      0.008259051479399204,
      0.01045436505228281,
      0.004781046882271767,
      0.0038169848266988993,
      0.00973602756857872,
      0.0007351008825935423,
      4.601275804816396e-08
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_",
      "pos",
      "(",
      "n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ],
    "token": "ums"
  },
  "9": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Exceptions"
    ],
    "probabilities": [
      0.0008184036705642939,
      0.01597767136991024,
      0.026960037648677826,
      0.021135279908776283,
      0.0175229050219059,
      0.02715529128909111,
      0.0024585300125181675,
      0.021291503682732582,
      5.040129202882326e-09
    ],
    "rationales": [
      "def",
      " take",
      "_",
      "while",
      "_",
      "pos",
      "(",
      "n",
      "ums"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    "token": "):"
  },
  "_phrase": "def take_while_pos(nums):\n    res = []\n    for n in nums:\n        if n <= 0:\n            break\n        res.append(n)\n    return res\n\ndef take_while_"
}
curl -X POST http://127.0.0.1:5000/prompt -H "Content-Type: application/json"  0.02s user 0.02s system 0% cpu 1:02.41 total


Sample 10

time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def get_nested(d, path, default=None):\n    cur = d\n    for k in path:\n        if k not in cur:\n            return default\n        cur = cur[k]\n    return cur\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.0005740290507674217
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " get"
  },
  "10": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      4.401558908284642e-05,
      0.005444006994366646,
      0.004620769526809454,
      0.006273471284657717,
      0.006013318430632353,
      0.003852903377264738,
      0.005075229797512293,
      0.0037389385979622602,
      0.0052212984301149845,
      2.448372811159061e-07
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ","
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9
    ],
    "token": " default"
  },
  "11": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.00552400341257453,
      0.06496217101812363,
      0.07009083777666092,
      0.045969441533088684,
      0.08037430047988892,
      0.05969301238656044,
      0.08993992954492569,
      0.01711837202310562,
      0.04665970057249069,
      0.04516396299004555,
      6.333678720382707e-11
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10
    ],
    "token": "="
  },
  "12": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      9.7042960987892e-05,
      0.12395419180393219,
      0.062308792024850845,
      0.01173405721783638,
      4.04784472607389e-09
    ],
    "rationales": [
      "def",
      " path",
      ",",
      " default",
      "="
    ],
    "rationales_indexes": [
      0,
      8,
      9,
      10,
      11
    ],
    "token": "None"
  },
  "13": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Functional"
    ],
    "probabilities": [
      3.407405529287644e-05,
      0.04900940880179405,
      0.07654278725385666,
      0.009349497966468334,
      0.06821145862340927,
      0.020444583147764206,
      0.0070727672427892685,
      0.0014332692371681333,
      0.08053452521562576,
      0.08010397851467133,
      0.0705186277627945,
      0.05750995874404907,
      7.985581760294735e-07
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default",
      "=",
      "None"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12
    ],
    "token": "):"
  },
  "14": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.39806830883026123,
      0.2861264646053314,
      0.058278586715459824
    ],
    "rationales": [
      "ested",
      " default",
      "):"
    ],
    "rationales_indexes": [
      4,
      10,
      13
    ],
    "token": "\n"
  },
  "15": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      2.0354098069219617e-06,
      0.0009805923327803612,
      2.4214230506913736e-05,
      0.0006988747627474368,
      0.00017880264203995466,
      1.875469934020657e-06,
      0.0008724749786779284,
      1.8900271243182942e-05,
      8.215972047764808e-05,
      0.00042763023520819843,
      0.0023132923524826765,
      8.249995880760252e-05,
      0.0028987484984099865,
      3.666200063889846e-05,
      2.9265189382954304e-08
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14
    ],
    "token": " "
  },
  "16": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9994900226593018
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      15
    ],
    "token": " "
  },
  "17": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.9994862079620361
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      16
    ],
    "token": " "
  },
  "18": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      3.824365649052197e-06,
      9.413572115590796e-06,
      5.902442353544757e-05,
      4.652138886740431e-05,
      3.610408020904288e-05,
      1.8326132703805342e-05,
      0.0001935022446559742,
      0.00020590986241586506,
      0.00015010981587693095,
      0.00023326555674429983,
      0.00022167860879562795,
      9.610333654563874e-05,
      0.00022576488845515996,
      8.358616469195113e-05,
      2.832625432347413e-05,
      2.6285386411473155e-05,
      0.0001383168128086254,
      4.3313534980882196e-09
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17
    ],
    "token": " cur"
  },
  "19": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.009262082166969776,
      0.12582211196422577,
      3.402574402788794e-11
    ],
    "rationales": [
      "def",
      "):",
      " cur"
    ],
    "rationales_indexes": [
      0,
      13,
      18
    ],
    "token": " ="
  },
  "2": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "OOP"
    ],
    "probabilities": [
      0.2635762393474579,
      1.1839746072439539e-08
    ],
    "rationales": [
      "def",
      " get"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "_"
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0031932408455759287,
      0.10442297905683517,
      0.0388544462621212,
      0.005511094816029072,
      0.11791029572486877,
      0.0019343841122463346,
      0.005579514428973198,
      0.014387951232492924,
      0.006449106615036726,
      0.08682692795991898,
      0.01442104484885931,
      0.005823853891342878,
      0.00013939484779257327
    ],
    "rationales": [
      "def",
      "_",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default",
      "=",
      "):",
      " ",
      " ",
      " ="
    ],
    "rationales_indexes": [
      0,
      2,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      13,
      15,
      16,
      19
    ],
    "token": " d"
  },
  "21": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.28860217332839966,
      0.1511833667755127,
      1.1718855219555735e-08
    ],
    "rationales": [
      "):",
      "\n",
      " d"
    ],
    "rationales_indexes": [
      13,
      14,
      20
    ],
    "token": "\n"
  },
  "22": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0004090099537279457,
      0.2301495373249054,
      0.0712011530995369,
      0.45018333196640015,
      0.0033854038920253515,
      0.0018111715326085687,
      0.07413575053215027,
      1.2532242976703856e-07
    ],
    "rationales": [
      " get",
      " default",
      "None",
      " ",
      " ",
      " cur",
      " =",
      "\n"
    ],
    "rationales_indexes": [
      1,
      10,
      12,
      16,
      17,
      18,
      19,
      21
    ],
    "token": " "
  },
  "23": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      0.999372661113739
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      22
    ],
    "token": " "
  },
  "24": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9994456171989441
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      23
    ],
    "token": " "
  },
  "25": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.0005238916492089629,
      0.010578470304608345,
      0.01757795363664627,
      0.013718586415052414,
      0.02081884630024433,
      0.0003954059211537242,
      0.004909956827759743,
      0.00831605400890112,
      0.0064315395429730415,
      0.004413333255797625,
      0.0037332731299102306,
      0.004883722402155399,
      0.018539687618613243,
      0.001874658395536244,
      0.004345448222011328,
      0.007804246619343758,
      0.009077062830328941,
      0.004729165695607662,
      0.0044789668172597885,
      0.001569426036439836,
      0.021004192531108856,
      0.004394547548145056,
      0.0023181394208222628,
      0.002817004919052124,
      1.7312164857230528e-07
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " cur",
      " =",
      " d",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24
    ],
    "token": " for"
  },
  "26": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Signature"
    ],
    "probabilities": [
      0.0005746331298723817,
      0.08750695735216141,
      0.025308983400464058,
      0.03720694035291672,
      0.09285444021224976,
      0.05172385647892952,
      0.06102707237005234,
      0.09463975578546524,
      0.03507109358906746,
      0.08846793323755264,
      0.09278804808855057,
      0.08885593712329865,
      0.09101931750774384,
      0.004993039648979902,
      0.09128893166780472,
      0.04348670691251755,
      0.08925867825746536,
      0.09035175293684006,
      0.0962844267487526,
      0.013574767857789993,
      0.07559342682361603,
      0.046162478625774384,
      0.07034273445606232,
      0.0574960857629776,
      0.05240896716713905,
      1.9953072296630125e-06
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " cur",
      " =",
      " d",
      "\n",
      " ",
      " ",
      " ",
      " for"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25
    ],
    "token": " k"
  },
  "27": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.10499683022499084,
      0.059608183801174164,
      0.6503691077232361,
      0.031556278467178345,
      0.032395943999290466,
      0.00689243758097291,
      2.8090532850910677e-06
    ],
    "rationales": [
      "def",
      "_",
      "):",
      " =",
      " ",
      " for",
      " k"
    ],
    "rationales_indexes": [
      0,
      2,
      13,
      19,
      22,
      25,
      26
    ],
    "token": " in"
  },
  "28": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.01873093843460083,
      0.13826502859592438,
      0.11594407260417938,
      0.11477123200893402,
      0.12415504455566406,
      0.07217764109373093,
      0.12007112801074982,
      0.007556073367595673,
      0.005573243368417025,
      0.09215078502893448,
      0.11905231326818466,
      0.0015098138246685266,
      0.0004210136830806732,
      0.0005246359505690634,
      0.00036563968751579523,
      0.000595489633269608,
      0.00016634112398605794,
      0.000422148616053164,
      0.0003345345612615347,
      0.0008028492447920144,
      0.0008512953063473105,
      0.003325483063235879,
      0.0004253270453773439,
      0.00035272157401777804,
      0.0001423941139364615,
      0.0005498790997080505,
      0.0008092365460470319,
      5.697431788576068e-06
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " cur",
      " =",
      " d",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " k",
      " in"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27
    ],
    "token": " path"
  },
  "29": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.048590127378702164,
      0.01564313843846321,
      0.03769458085298538,
      0.01690763048827648,
      0.023317158222198486,
      0.009005244821310043,
      0.022367317229509354,
      0.013372678309679031,
      0.013732107356190681,
      0.030807675793766975,
      0.026524968445301056,
      0.1618083417415619,
      0.018452651798725128,
      5.23565857335484e-10
    ],
    "rationales": [
      "def",
      "=",
      "None",
      "):",
      "\n",
      " =",
      " d",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " in",
      " path"
    ],
    "rationales_indexes": [
      0,
      11,
      12,
      13,
      14,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      27,
      28
    ],
    "token": ":"
  },
  "3": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.0052802665159106255,
      0.002969511551782489,
      4.01479428546736e-06
    ],
    "rationales": [
      "def",
      " get",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "n"
  },
  "30": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.05877182260155678,
      0.11447830498218536,
      0.29717403650283813,
      0.046992309391498566,
      0.3317500650882721,
      0.2921048104763031,
      0.4570119082927704,
      0.18265101313591003,
      0.2010101079940796,
      0.05950196087360382,
      4.491244453674881e-06
    ],
    "rationales": [
      "_",
      "d",
      "=",
      "):",
      " cur",
      " =",
      " ",
      " ",
      " for",
      " in",
      ":"
    ],
    "rationales_indexes": [
      2,
      6,
      11,
      13,
      18,
      19,
      23,
      24,
      25,
      27,
      29
    ],
    "token": "\n"
  },
  "31": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Indentation"
    ],
    "probabilities": [
      0.011085564270615578,
      0.00962920393794775,
      0.8920281529426575,
      0.22924189269542694,
      2.9933983114460716e-07
    ],
    "rationales": [
      " get",
      " default",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      1,
      10,
      23,
      24,
      30
    ],
    "token": " "
  },
  "32": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9992952346801758
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      31
    ],
    "token": " "
  },
  "33": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.9994258880615234
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      32
    ],
    "token": " "
  },
  "34": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9992619156837463
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      33
    ],
    "token": " "
  },
  "35": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9993482232093811
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      34
    ],
    "token": " "
  },
  "36": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9992576241493225
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      35
    ],
    "token": " "
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9992762207984924
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      36
    ],
    "token": " "
  },
  "38": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.0008475988870486617,
      0.06959806382656097,
      0.08889811486005783,
      0.11237463355064392,
      0.12241415679454803,
      0.0004715850518550724,
      0.016352077946066856,
      0.014887699857354164,
      0.007955175824463367,
      0.017914842814207077,
      0.022769087925553322,
      0.07936704158782959,
      0.01587373949587345,
      0.06071751192212105,
      0.012454674579203129,
      0.04832252487540245,
      0.13507941365242004,
      0.10170754790306091,
      0.0949220284819603,
      0.004624335560947657,
      0.013251068070530891,
      0.11220282316207886,
      0.0016095206374302506,
      0.028297940269112587,
      0.02461925707757473,
      0.024039585143327713,
      0.12953068315982819,
      0.012532474473118782,
      0.006556655280292034,
      0.0906561091542244,
      0.005795877426862717,
      0.00977097637951374,
      0.01233472116291523,
      0.019656559452414513,
      0.023257609456777573,
      0.02594374120235443,
      0.012256959453225136,
      1.2833191931349575e-07
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " cur",
      " =",
      " d",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " k",
      " in",
      " path",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37
    ],
    "token": " if"
  },
  "39": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.027379926294088364,
      0.46539390087127686,
      0.09590019285678864,
      1.2509738667176862e-07
    ],
    "rationales": [
      " k",
      " path",
      ":",
      " if"
    ],
    "rationales_indexes": [
      26,
      28,
      29,
      38
    ],
    "token": " k"
  },
  "4": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.00019307740149088204,
      0.057488877326250076,
      0.016639098525047302,
      1.5034630207466648e-09
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "ested"
  },
  "40": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.004852525889873505,
      0.07357566058635712,
      0.07103386521339417,
      4.767874202116218e-07
    ],
    "rationales": [
      " get",
      "_",
      "ested",
      " k"
    ],
    "rationales_indexes": [
      1,
      2,
      4,
      39
    ],
    "token": " not"
  },
  "41": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.3940139710903168,
      0.001509434194304049
    ],
    "rationales": [
      " in",
      " not"
    ],
    "rationales_indexes": [
      27,
      40
    ],
    "token": " in"
  },
  "42": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.4350530207157135,
      0.008880939334630966,
      6.86964049236849e-05,
      0.011069332249462605,
      1.657102302488056e-06
    ],
    "rationales": [
      " get",
      "(",
      " cur",
      "\n",
      " in"
    ],
    "rationales_indexes": [
      1,
      5,
      18,
      21,
      41
    ],
    "token": " cur"
  },
  "43": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.1631537824869156,
      0.0031482321210205555,
      7.497161919056694e-11
    ],
    "rationales": [
      "None",
      "):",
      " cur"
    ],
    "rationales_indexes": [
      12,
      13,
      42
    ],
    "token": ":"
  },
  "44": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.337887704372406,
      0.057836540043354034,
      3.671718104669708e-06
    ],
    "rationales": [
      "ested",
      "\n",
      ":"
    ],
    "rationales_indexes": [
      4,
      30,
      43
    ],
    "token": "\n"
  },
  "45": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.4290194511413574,
      0.17084212601184845,
      0.7090946435928345,
      0.05757497623562813,
      0.002965440507978201,
      2.0230361030826316e-07
    ],
    "rationales": [
      " ",
      " ",
      " ",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      23,
      34,
      35,
      36,
      37,
      44
    ],
    "token": " "
  },
  "46": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.9994580149650574
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      45
    ],
    "token": " "
  },
  "47": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.9993767142295837
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      46
    ],
    "token": " "
  },
  "48": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.999299168586731
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      47
    ],
    "token": " "
  },
  "49": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "With"
    ],
    "probabilities": [
      0.9995001554489136
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      48
    ],
    "token": " "
  },
  "5": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.003241347847506404,
      0.02015593834221363,
      0.04820631816983223,
      0.040960513055324554,
      6.733202462783083e-05
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "("
  },
  "50": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.9993146657943726
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      49
    ],
    "token": " "
  },
  "51": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Statements"
    ],
    "probabilities": [
      0.999398946762085
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      50
    ],
    "token": " "
  },
  "52": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9992628693580627
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      51
    ],
    "token": " "
  },
  "53": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "With"
    ],
    "probabilities": [
      0.9992802739143372
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      52
    ],
    "token": " "
  },
  "54": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.9994105100631714
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      53
    ],
    "token": " "
  },
  "55": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9993746876716614
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      54
    ],
    "token": " "
  },
  "56": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      3.723945337696932e-05,
      0.026382293552160263,
      0.06640560925006866,
      0.008410357870161533,
      0.0002981983416248113,
      0.0023465848062187433,
      1.7489476533683046e-10
    ],
    "rationales": [
      "def",
      " get",
      "None",
      "):",
      " if",
      " in",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      12,
      13,
      38,
      41,
      55
    ],
    "token": " return"
  },
  "57": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.13223543763160706,
      0.34200939536094666,
      0.18769966065883636,
      0.39508873224258423,
      0.26187288761138916,
      0.004625333938747644,
      0.011702317744493484,
      0.018597211688756943,
      4.519159483606927e-05
    ],
    "rationales": [
      "=",
      "None",
      " d",
      " ",
      " k",
      "\n",
      " ",
      " not",
      " return"
    ],
    "rationales_indexes": [
      11,
      12,
      20,
      24,
      26,
      30,
      31,
      40,
      56
    ],
    "token": " default"
  },
  "58": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9038624167442322,
      3.0670129547161196e-08
    ],
    "rationales": [
      "\n",
      " default"
    ],
    "rationales_indexes": [
      44,
      57
    ],
    "token": "\n"
  },
  "59": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.23022495210170746,
      0.06154210865497589,
      0.5076209902763367,
      4.598754443918551e-08
    ],
    "rationales": [
      " ",
      " ",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      54,
      55,
      56,
      58
    ],
    "token": " "
  },
  "6": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.003098048036918044,
      0.00519568333402276,
      0.006414845585823059,
      0.005124847404658794,
      0.006287731230258942,
      0.00013803747424390167
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "token": "d"
  },
  "60": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9994662404060364
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      59
    ],
    "token": " "
  },
  "61": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9994087219238281
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      60
    ],
    "token": " "
  },
  "62": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.99940025806427
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      61
    ],
    "token": " "
  },
  "63": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Asserts"
    ],
    "probabilities": [
      0.9993459582328796
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      62
    ],
    "token": " "
  },
  "64": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9994564652442932
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      63
    ],
    "token": " "
  },
  "65": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.9994423985481262
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      64
    ],
    "token": " "
  },
  "66": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      6.879346983623691e-06,
      0.00031390762887895107,
      0.023270143195986748,
      2.1395440853666514e-05,
      0.06660214066505432,
      0.0001202296043629758,
      9.100206144196932e-10
    ],
    "rationales": [
      "def",
      " get",
      " ",
      " cur",
      " =",
      " return",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      17,
      18,
      19,
      56,
      65
    ],
    "token": " cur"
  },
  "67": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.0012834479566663504,
      0.6537670493125916,
      0.2270917445421219,
      0.09080788493156433,
      4.938753225824932e-11
    ],
    "rationales": [
      "def",
      " path",
      " cur",
      " =",
      " cur"
    ],
    "rationales_indexes": [
      0,
      8,
      18,
      19,
      66
    ],
    "token": " ="
  },
  "68": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.6170147061347961,
      0.0012737588258460164,
      1.7031571708514548e-09
    ],
    "rationales": [
      " ",
      " cur",
      " ="
    ],
    "rationales_indexes": [
      64,
      66,
      67
    ],
    "token": " cur"
  },
  "69": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.0006172687280923128,
      0.20898951590061188,
      0.12534798681735992,
      0.007068675011396408,
      0.15314801037311554,
      0.03542415052652359,
      0.011734114028513432,
      0.007965817116200924,
      0.1023048684000969,
      0.11372563987970352,
      0.010285548865795135,
      0.05408907309174538,
      0.004413086920976639,
      0.0026829459238797426,
      0.11647723615169525,
      0.17502331733703613,
      0.1647747904062271,
      0.19786643981933594,
      0.012676787562668324,
      0.020614758133888245,
      0.09095585346221924,
      0.1224164366722107,
      0.10135823488235474,
      0.10142456740140915,
      0.10329647362232208,
      0.010832936502993107,
      0.10404402762651443,
      0.07861192524433136,
      0.09760219603776932,
      0.11774083226919174,
      0.11838633567094803,
      0.20857460796833038,
      0.21329019963741302,
      0.20190201699733734,
      0.1941269189119339,
      0.1887446641921997,
      0.18473517894744873,
      0.18019917607307434,
      0.1120096743106842,
      0.09781692922115326,
      0.11496836692094803,
      0.005975259002298117,
      0.14786188304424286,
      0.09794723987579346,
      0.11829443275928497,
      0.19133195281028748,
      0.18798212707042694,
      0.18415702879428864,
      0.1808440238237381,
      0.17740578949451447,
      0.17315976321697235,
      0.16839571297168732,
      0.16417276859283447,
      0.15712642669677734,
      0.1590023785829544,
      0.16114088892936707,
      0.12308254837989807,
      0.11414219439029694,
      0.00890770647674799,
      0.20404288172721863,
      0.1473785787820816,
      0.14151419699192047,
      0.15807649493217468,
      0.12997427582740784,
      0.1083739772439003,
      0.07113748043775558,
      0.051434654742479324,
      0.025004830211400986,
      4.0695767068721356e-13
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ",",
      " path",
      ",",
      " default",
      "=",
      "None",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " cur",
      " =",
      " d",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " k",
      " in",
      " path",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " if",
      " k",
      " not",
      " in",
      " cur",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " default",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " cur",
      " =",
      " cur"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      60,
      61,
      62,
      63,
      64,
      65,
      66,
      67,
      68
    ],
    "token": "["
  },
  "7": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.02134549804031849,
      0.13606242835521698,
      0.025991907343268394,
      6.614728044951335e-05
    ],
    "rationales": [
      "def",
      " get",
      "(",
      "d"
    ],
    "rationales_indexes": [
      0,
      1,
      5,
      6
    ],
    "token": ","
  },
  "70": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.005896462593227625,
      0.02645556442439556,
      0.031645774841308594,
      0.014049883931875229,
      0.016394656151533127,
      0.002747289603576064,
      0.008965179324150085,
      0.021307794377207756,
      0.008995444513857365,
      0.054253991693258286,
      0.05010848492383957,
      0.04375411570072174,
      0.0009309781016781926,
      0.031206946820020676,
      0.02928602136671543,
      0.012452833354473114,
      0.014280348084867,
      2.9264134354889393e-05
    ],
    "rationales": [
      "def",
      " get",
      "n",
      "d",
      ",",
      " default",
      "=",
      "):",
      " k",
      " ",
      " ",
      " if",
      " k",
      " not",
      ":",
      " return",
      " default",
      "["
    ],
    "rationales_indexes": [
      0,
      1,
      3,
      6,
      7,
      10,
      11,
      13,
      26,
      36,
      37,
      38,
      39,
      40,
      43,
      56,
      57,
      69
    ],
    "token": "k"
  },
  "71": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.08637634664773941,
      0.29449698328971863,
      0.33264440298080444,
      0.22496286034584045,
      0.1268485188484192,
      3.659273772882443e-07
    ],
    "rationales": [
      "def",
      " get",
      "n",
      " k",
      "[",
      "k"
    ],
    "rationales_indexes": [
      0,
      1,
      3,
      26,
      69,
      70
    ],
    "token": "]"
  },
  "72": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.202204167842865,
      0.010574263520538807
    ],
    "rationales": [
      "def",
      "]"
    ],
    "rationales_indexes": [
      0,
      71
    ],
    "token": "\n"
  },
  "73": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.5589519739151001,
      0.3354420065879822,
      0.16852392256259918,
      0.03840915486216545,
      9.318250171475029e-09
    ],
    "rationales": [
      " ",
      " ",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      48,
      63,
      64,
      65,
      72
    ],
    "token": " "
  },
  "74": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9994712471961975
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      73
    ],
    "token": " "
  },
  "75": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.9995054006576538
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      74
    ],
    "token": " "
  },
  "76": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      7.620348333148286e-05,
      0.048761144280433655,
      0.07000737637281418,
      0.060339223593473434,
      0.08054225146770477,
      0.12604285776615143,
      0.08495412766933441,
      0.052943456918001175,
      0.017326703295111656,
      0.06654821336269379,
      0.029239043593406677,
      0.02321779727935791,
      0.04276654124259949,
      0.00654646847397089,
      0.07788162678480148,
      0.000244591647060588,
      0.002411495428532362,
      0.010310824029147625,
      4.7583249397575855e-05,
      0.0005706902593374252,
      1.1539317418263195e-10
    ],
    "rationales": [
      "def",
      " path",
      "None",
      " cur",
      " ",
      " ",
      " for",
      " k",
      " path",
      " ",
      " if",
      " k",
      " not",
      " cur",
      " ",
      " return",
      " =",
      " cur",
      "[",
      "]",
      " "
    ],
    "rationales_indexes": [
      0,
      8,
      12,
      18,
      22,
      24,
      25,
      26,
      28,
      32,
      38,
      39,
      40,
      42,
      45,
      56,
      67,
      68,
      69,
      71,
      75
    ],
    "token": " return"
  },
  "77": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      9.626160317566246e-05,
      0.03258556127548218,
      0.0035010124556720257,
      0.11115200817584991,
      5.278762760241307e-09
    ],
    "rationales": [
      "def",
      " get",
      " cur",
      " =",
      " return"
    ],
    "rationales_indexes": [
      0,
      1,
      18,
      19,
      76
    ],
    "token": " cur"
  },
  "78": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.8358948230743408,
      3.648614965712227e-09
    ],
    "rationales": [
      "\n",
      " cur"
    ],
    "rationales_indexes": [
      58,
      77
    ],
    "token": "\n"
  },
  "79": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Signature"
    ],
    "probabilities": [
      0.9996256828308105
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      78
    ],
    "token": "\n"
  },
  "8": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      8.547216566512361e-05,
      0.0005631555686704814,
      0.002678609685972333,
      0.003888905979692936,
      0.0023506865836679935,
      0.001479402999393642,
      0.0016236562514677644,
      1.820738248170528e-06
    ],
    "rationales": [
      "def",
      " get",
      "_",
      "n",
      "ested",
      "(",
      "d",
      ","
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ],
    "token": " path"
  },
  "9": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.6374080777168274,
      0.19453613460063934,
      0.021467160433530807,
      1.3703526065000915e-06
    ],
    "rationales": [
      "n",
      "d",
      ",",
      " path"
    ],
    "rationales_indexes": [
      3,
      6,
      7,
      8
    ],
    "token": ","
  },
  "_phrase": "def get_nested(d, path, default=None):\n    cur = d\n    for k in path:\n        if k not in cur:\n            return default\n        cur = cur[k]\n    return cur\n\n"
}
curl -X POST http://127.0.0.1:5000/prompt -H "Content-Type: application/json"  0.02s user 0.02s system 0% cpu 1:22.64 total



Sample 11

curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def invert_map(m):\n    out = {}\n    for k, v in m.items():\n        out[v] = k\n    return out\n"}'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.004643934778869152
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " in"
  },
  "10": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.9994294047355652
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      9
    ],
    "token": " "
  },
  "11": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.999504566192627
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      10
    ],
    "token": " "
  },
  "12": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      3.208129055565223e-05,
      9.12902396521531e-05,
      0.00015521918248850852,
      2.3639593564439565e-05,
      0.0002165566838812083,
      2.131035944330506e-05,
      0.00019278796389698982,
      0.00022496013843920082,
      0.0011316909221932292,
      0.0004207781166769564,
      8.40076245367527e-05,
      1.745732980396042e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11
    ],
    "token": " out"
  },
  "13": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Functional"
    ],
    "probabilities": [
      7.628959428984672e-05,
      0.08125164359807968,
      0.20127242803573608,
      3.882590959847221e-08
    ],
    "rationales": [
      "def",
      "):",
      "\n",
      " out"
    ],
    "rationales_indexes": [
      0,
      7,
      8,
      12
    ],
    "token": " ="
  },
  "14": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.0008082770509645343,
      0.0023207226768136024,
      0.0024328669533133507,
      0.005156738217920065,
      0.004494716878980398,
      0.0053999461233615875,
      0.003760020947083831,
      0.003360687755048275,
      0.004858205560594797,
      0.004657674580812454,
      0.004899024963378906,
      0.0046469480730593204,
      0.004808308091014624,
      1.5666330455132993e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " ="
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13
    ],
    "token": " {}"
  },
  "15": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.30411338806152344,
      5.6970959121827036e-05
    ],
    "rationales": [
      "):",
      " {}"
    ],
    "rationales_indexes": [
      7,
      14
    ],
    "token": "\n"
  },
  "16": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.002940824721008539,
      0.3361113965511322,
      3.132145209860937e-08
    ],
    "rationales": [
      " ",
      " =",
      "\n"
    ],
    "rationales_indexes": [
      11,
      13,
      15
    ],
    "token": " "
  },
  "17": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.9994862079620361
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      16
    ],
    "token": " "
  },
  "18": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Asserts"
    ],
    "probabilities": [
      0.9994465708732605
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      17
    ],
    "token": " "
  },
  "19": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.000744360382668674,
      0.0945507064461708,
      0.0015791154000908136,
      0.1040189117193222,
      0.11701187491416931,
      0.00030885133310221136,
      0.0020062667317688465,
      0.04276895523071289,
      0.0011932294582948089,
      0.0008443064871244133,
      0.06683024019002914,
      0.002113957656547427,
      0.004409159068018198,
      0.0028162517119199038,
      0.011830441653728485,
      0.0025281235575675964,
      0.002341565676033497,
      0.0031922217458486557,
      1.4283772031831177e-07
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18
    ],
    "token": " for"
  },
  "2": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0001328982471022755,
      2.1865679839666585e-12
    ],
    "rationales": [
      "def",
      " in"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "vert"
  },
  "20": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0005359220667742193,
      0.06874440610408783,
      0.04714268445968628,
      0.048507824540138245,
      0.048284128308296204,
      0.06750740110874176,
      0.036193713545799255,
      0.060286328196525574,
      0.06754721701145172,
      0.06639983505010605,
      0.06150505319237709,
      0.05147179588675499,
      0.02545998990535736,
      0.008353379555046558,
      0.03358158841729164,
      0.04551726579666138,
      0.05807099863886833,
      0.05443081632256508,
      0.055659856647253036,
      3.6085125429963227e-06
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " ",
      " for"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19
    ],
    "token": " k"
  },
  "21": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.05337857827544212,
      0.06544661521911621,
      0.05365435406565666,
      0.06318067759275436,
      0.0714430958032608,
      0.48930981755256653,
      0.284047394990921,
      0.1066080704331398,
      1.5387853636639193e-05
    ],
    "rationales": [
      "def",
      "vert",
      "_",
      "):",
      " out",
      " =",
      " ",
      " for",
      " k"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      7,
      12,
      13,
      18,
      19,
      20
    ],
    "token": ","
  },
  "22": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00023455919290427119,
      0.08421050757169724,
      0.03233680874109268,
      0.00718175433576107,
      0.002347260946407914,
      0.0014870527666062117,
      4.814361545868451e-06
    ],
    "rationales": [
      "def",
      "):",
      " =",
      " ",
      " for",
      " k",
      ","
    ],
    "rationales_indexes": [
      0,
      7,
      13,
      18,
      19,
      20,
      21
    ],
    "token": " v"
  },
  "23": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.13828237354755402,
      0.025381973013281822,
      0.0175801832228899,
      0.014389775693416595,
      0.007423491217195988,
      6.403104180208175e-07
    ],
    "rationales": [
      "def",
      "(",
      "):",
      " {}",
      " for",
      " v"
    ],
    "rationales_indexes": [
      0,
      5,
      7,
      14,
      19,
      22
    ],
    "token": " in"
  },
  "24": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.0009837507968768477,
      0.0712483748793602,
      0.004469496197998524,
      0.12820665538311005,
      0.2031295746564865,
      6.511116225738078e-05
    ],
    "rationales": [
      "def",
      " in",
      "m",
      " ",
      " k",
      " in"
    ],
    "rationales_indexes": [
      0,
      1,
      6,
      10,
      20,
      23
    ],
    "token": " m"
  },
  "25": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.04216836392879486,
      0.16170543432235718,
      0.03932971879839897,
      0.0549863800406456,
      0.04547853395342827,
      0.02970646694302559,
      0.0358988493680954,
      0.030591530725359917,
      0.04418950900435448,
      1.7822168274506112e-07
    ],
    "rationales": [
      "def",
      "_",
      "m",
      " out",
      " for",
      " k",
      ",",
      " v",
      " in",
      " m"
    ],
    "rationales_indexes": [
      0,
      3,
      6,
      12,
      19,
      20,
      21,
      22,
      23,
      24
    ],
    "token": "."
  },
  "26": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      1.0577382454357576e-05,
      0.0002999488788191229,
      0.01779025048017502,
      0.03174738585948944,
      0.0001111763485823758,
      0.007302064914256334,
      0.0008795924950391054,
      1.3446843638575956e-08
    ],
    "rationales": [
      "def",
      "_",
      "map",
      "(",
      "):",
      " for",
      " in",
      "."
    ],
    "rationales_indexes": [
      0,
      3,
      4,
      5,
      7,
      19,
      23,
      25
    ],
    "token": "items"
  },
  "27": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00012925370538141578,
      0.44555848836898804,
      0.00248103984631598,
      0.0009109788225032389,
      0.06143005192279816,
      0.005045786499977112,
      0.013931289315223694,
      7.60669394139768e-09
    ],
    "rationales": [
      "def",
      "_",
      "m",
      "):",
      " for",
      " in",
      ".",
      "items"
    ],
    "rationales_indexes": [
      0,
      3,
      6,
      7,
      19,
      23,
      25,
      26
    ],
    "token": "():"
  },
  "28": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.7252012491226196,
      5.607935236184858e-05
    ],
    "rationales": [
      "\n",
      "():"
    ],
    "rationales_indexes": [
      15,
      27
    ],
    "token": "\n"
  },
  "29": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.4665857255458832,
      0.6970933079719543,
      0.1275458037853241,
      0.01514727994799614,
      0.00015423486183863133,
      2.8258369866307476e-07
    ],
    "rationales": [
      " ",
      " =",
      " ",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      10,
      13,
      16,
      17,
      18,
      28
    ],
    "token": " "
  },
  "3": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.049718860536813736,
      0.08333893865346909,
      1.4042745632991682e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "_"
  },
  "30": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9993077516555786
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      29
    ],
    "token": " "
  },
  "31": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.9993191957473755
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      30
    ],
    "token": " "
  },
  "32": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9992952346801758
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      31
    ],
    "token": " "
  },
  "33": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.9994258880615234
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      32
    ],
    "token": " "
  },
  "34": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9992619156837463
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      33
    ],
    "token": " "
  },
  "35": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9993482232093811
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      34
    ],
    "token": " "
  },
  "36": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      1.2821828022424597e-05,
      0.012008055113255978,
      0.030686737969517708,
      0.032044824212789536,
      0.006452462170273066,
      0.0319024957716465,
      0.017542436718940735,
      0.007171932607889175,
      0.022599073126912117,
      0.007596177514642477,
      0.0046691601164639,
      0.0026513852644711733,
      5.836285345139913e-05,
      0.016830191016197205,
      3.0041335776331834e-05,
      0.019553575664758682,
      0.01924770697951317,
      0.02092747576534748,
      0.032037340104579926,
      0.020259113982319832,
      0.029505455866456032,
      2.1611203919746913e-05,
      4.4300548324827105e-05,
      0.0009630157728679478,
      2.5102122890530154e-05,
      0.0029600574634969234,
      0.011658593080937862,
      3.9564551116200164e-05,
      0.001475475961342454,
      0.0003140124026685953,
      0.0001053667874657549,
      0.0007547980058006942,
      1.9650613467092626e-05,
      0.0013440856710076332,
      0.0018687908304855227,
      4.02741449079258e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " k",
      ",",
      " v",
      " in",
      " m",
      ".",
      "items",
      "():",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35
    ],
    "token": " out"
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.00013292237417772412,
      0.011900685727596283,
      0.232132226228714,
      0.007591916248202324,
      0.023323697969317436,
      0.12098576128482819,
      0.1529783308506012,
      0.0028360954020172358,
      0.08909974247217178,
      0.16069310903549194,
      0.1526115983724594,
      0.1974925845861435,
      0.03629600256681442,
      0.01835571974515915,
      0.06356427818536758,
      0.25299206376075745,
      0.17580334842205048,
      0.182384192943573,
      0.1912044882774353,
      0.25533849000930786,
      0.17417801916599274,
      0.16635474562644958,
      0.260482519865036,
      0.2524064779281616,
      0.24851222336292267,
      0.07848816365003586,
      0.24035505950450897,
      0.2148548662662506,
      0.21895462274551392,
      0.22711512446403503,
      0.22500158846378326,
      0.22111892700195312,
      0.2116222232580185,
      0.18008682131767273,
      0.12983043491840363,
      0.10345269739627838,
      6.911163268341625e-07
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " k",
      ",",
      " v",
      " in",
      " m",
      ".",
      "items",
      "():",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " out"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36
    ],
    "token": "["
  },
  "38": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.0015499723376706243,
      0.046976979821920395,
      0.018821682780981064,
      0.07352974265813828,
      0.0307348370552063,
      0.010044409893453121,
      5.448785486805718e-06
    ],
    "rationales": [
      "def",
      "m",
      "):",
      " k",
      ",",
      " v",
      "["
    ],
    "rationales_indexes": [
      0,
      6,
      7,
      20,
      21,
      22,
      37
    ],
    "token": "v"
  },
  "39": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.539456844329834,
      0.16760021448135376,
      1.6001536096155178e-07
    ],
    "rationales": [
      " out",
      "[",
      "v"
    ],
    "rationales_indexes": [
      36,
      37,
      38
    ],
    "token": "]"
  },
  "4": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.0019030083203688264,
      0.0029502855613827705,
      0.0018593764398247004,
      9.793885169528949e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "map"
  },
  "40": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0036984614562243223,
      0.03566648066043854,
      0.24974505603313446,
      0.00014856824418529868
    ],
    "rationales": [
      "def",
      " =",
      "[",
      "]"
    ],
    "rationales_indexes": [
      0,
      13,
      37,
      39
    ],
    "token": " ="
  },
  "41": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Statements"
    ],
    "probabilities": [
      0.0005403705290518701,
      0.038557518273591995,
      0.04668039828538895,
      0.015278986655175686,
      0.10347408801317215,
      0.02807946503162384,
      0.02191704884171486,
      0.040933385491371155,
      0.06478850543498993,
      0.03481868654489517,
      0.08837969601154327,
      0.008982685394585133,
      0.05409364402294159,
      1.9055884195040562e-06
    ],
    "rationales": [
      "def",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " {}",
      " ",
      " k",
      ".",
      " ="
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      11,
      14,
      18,
      20,
      25,
      40
    ],
    "token": " k"
  },
  "42": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.23339831829071045,
      7.0017749749240465e-06
    ],
    "rationales": [
      " =",
      " k"
    ],
    "rationales_indexes": [
      40,
      41
    ],
    "token": "\n"
  },
  "43": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.11968293786048889,
      0.9676706194877625,
      0.0031396362464874983,
      1.7619690595438442e-07
    ],
    "rationales": [
      "():",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      27,
      30,
      35,
      42
    ],
    "token": " "
  },
  "44": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9993201494216919
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      43
    ],
    "token": " "
  },
  "45": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9993821382522583
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      44
    ],
    "token": " "
  },
  "46": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.0005735348677262664,
      0.06359754502773285,
      0.00023117601813282818,
      0.04623054713010788,
      0.0006697351927869022,
      0.04288335517048836,
      0.00014329208352137357,
      0.03631835803389549,
      0.0005031678010709584,
      0.05738892778754234,
      0.050135254859924316,
      0.0007319389842450619,
      0.0006277281208895147,
      0.0006553889834322035,
      0.039181191474199295,
      0.06721752136945724,
      0.0006288556614890695,
      0.032827261835336685,
      0.00018605927471071482,
      0.026025619357824326,
      0.0647859126329422,
      0.0005777011974714696,
      0.07378644496202469,
      0.00010942455264739692,
      0.07471545040607452,
      0.06770730763673782,
      0.0006434098468162119,
      7.709614146733657e-05,
      0.010117716155946255,
      0.0007172901532612741,
      0.0007170590688474476,
      0.0006573807331733406,
      0.0005393070168793201,
      0.0003819090488832444,
      0.00013108378334436566,
      5.082256029709242e-05,
      0.05850696936249733,
      4.1043102100957185e-05,
      0.00017781245696824044,
      0.012311731465160847,
      0.0002009596355492249,
      0.00011911686306120828,
      0.0029179707635194063,
      8.909777534427121e-05,
      0.07351817935705185,
      2.1768023494850297e-10
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " out",
      " =",
      " {}",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " k",
      ",",
      " v",
      " in",
      " m",
      ".",
      "items",
      "():",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " out",
      "[",
      "v",
      "]",
      " =",
      " k",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45
    ],
    "token": " return"
  },
  "47": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Bool"
    ],
    "probabilities": [
      0.000828124932013452,
      0.11929982155561447,
      6.19405818724772e-06
    ],
    "rationales": [
      " out",
      "\n",
      " return"
    ],
    "rationales_indexes": [
      36,
      42,
      46
    ],
    "token": " out"
  },
  "48": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.3323479890823364,
      1.4756413293071091e-05
    ],
    "rationales": [
      "\n",
      " out"
    ],
    "rationales_indexes": [
      42,
      47
    ],
    "token": "\n"
  },
  "49": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.9968153834342957
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      48
    ],
    "token": "\n"
  },
  "5": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.09929046779870987,
      0.06787649542093277,
      0.06354615837335587,
      0.07740329951047897,
      1.2816850869512564e-07
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "("
  },
  "6": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Statements"
    ],
    "probabilities": [
      0.002673191949725151,
      0.0073858448304235935,
      0.005867123603820801,
      0.010317721404135227,
      0.007397271692752838,
      9.167608368443325e-05
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "("
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "token": "m"
  },
  "7": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0008127083419822156,
      0.0015545116038993,
      0.0033387800212949514,
      0.0036388200242072344,
      0.002361996565014124,
      0.0017948491731658578,
      5.450808207574376e-11
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6
    ],
    "token": "):"
  },
  "8": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.8472586870193481,
      0.06658641248941422
    ],
    "rationales": [
      "vert",
      "):"
    ],
    "rationales_indexes": [
      2,
      7
    ],
    "token": "\n"
  },
  "9": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      1.835906004998833e-06,
      1.4819322586845374e-06,
      7.218062819447368e-06,
      2.6093657652381808e-06,
      1.7579928680788726e-05,
      5.1595852710306644e-05,
      1.249426986760227e-05,
      1.0562840543570928e-05,
      1.7788709172350536e-08
    ],
    "rationales": [
      "def",
      " in",
      "vert",
      "_",
      "map",
      "(",
      "m",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    "token": " "
  },
  "_phrase": "def invert_map(m):\n    out = {}\n    for k, v in m.items():\n        out[v] = k\n    return out\n\n"
}

Sample 11

time curl -X POST http://127.0.0.1:5000/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def normalize_scores(scores):\n    total = sum(scores)\n    if total == 0:\n        return [0 for _ in scores]\n    normalized = []\n    for s in scores:\n        normalized.append(s / total)\n    return normalized\n"
  }'

{
  "0": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [],
    "rationales": [],
    "rationales_indexes": [],
    "token": "def"
  },
  "1": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "With"
    ],
    "probabilities": [
      6.38582423562184e-05
    ],
    "rationales": [
      "def"
    ],
    "rationales_indexes": [
      0
    ],
    "token": " normal"
  },
  "10": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.6758778691291809,
      0.06292856484651566
    ],
    "rationales": [
      "ize",
      "):"
    ],
    "rationales_indexes": [
      2,
      9
    ],
    "token": "\n"
  },
  "100": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.02128227986395359,
      0.18626336753368378,
      7.2614487756084145e-09
    ],
    "rationales": [
      "def",
      " =",
      "score"
    ],
    "rationales_indexes": [
      0,
      15,
      99
    ],
    "token": " ="
  },
  "101": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0013248093891888857,
      0.22382338345050812,
      0.16466134786605835,
      0.1145310029387474,
      0.06895744800567627,
      0.09717247635126114,
      0.25261765718460083,
      0.00833280198276043,
      0.32291895151138306,
      0.17719395458698273,
      0.02476966194808483,
      0.1356905847787857,
      0.04455570876598358,
      0.013819932006299496,
      0.28660261631011963,
      0.0027668552938848734,
      0.00470730010420084,
      3.894439942087047e-05,
      2.260949472088214e-08
    ],
    "rationales": [
      "def",
      "(",
      " return",
      " for",
      " in",
      " for",
      "\n",
      "append",
      "(",
      "\n",
      "\n",
      "def",
      "init",
      " score",
      " ):",
      "self",
      " .",
      "score",
      " ="
    ],
    "rationales_indexes": [
      0,
      6,
      38,
      41,
      43,
      57,
      62,
      72,
      73,
      84,
      85,
      86,
      88,
      93,
      94,
      97,
      98,
      99,
      100
    ],
    "token": " score"
  },
  "102": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.20018388330936432,
      2.7085959573014406e-06
    ],
    "rationales": [
      "def",
      " score"
    ],
    "rationales_indexes": [
      0,
      101
    ],
    "token": "\n"
  },
  "103": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9988308548927307
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      102
    ],
    "token": "\n"
  },
  "104": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.00042624297202564776,
      0.08820264041423798,
      0.24613550305366516,
      0.1511387825012207,
      0.0036586918868124485,
      0.04170791059732437,
      0.020777549594640732,
      9.350462244761548e-12
    ],
    "rationales": [
      "def",
      "__",
      " ,",
      " score",
      " ):",
      "\n",
      "self",
      "\n"
    ],
    "rationales_indexes": [
      0,
      89,
      92,
      93,
      94,
      95,
      97,
      103
    ],
    "token": "def"
  },
  "105": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.0008825642289593816,
      0.2605213224887848,
      0.6436731815338135,
      8.877083956804199e-08
    ],
    "rationales": [
      "__",
      "\n",
      " score",
      "def"
    ],
    "rationales_indexes": [
      89,
      95,
      101,
      104
    ],
    "token": " __"
  },
  "106": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      1.910208993649576e-06,
      0.000572346558328718,
      0.00038546641007997096,
      0.0008357760380022228,
      0.0007977523491717875,
      0.02325264923274517,
      0.0008809666032902896,
      0.047312960028648376,
      0.03632057085633278,
      0.0009023884776979685,
      0.000683329242747277,
      0.03777296096086502,
      0.09500927478075027,
      0.09869197010993958,
      0.04450235515832901,
      0.039179157465696335,
      0.035146888345479965,
      0.0004034598241560161,
      0.07146694511175156,
      0.04147500544786453,
      0.09905163198709488,
      0.032826825976371765,
      0.11362916976213455,
      0.11828287690877914,
      0.1280992031097412,
      0.05315195769071579,
      0.12206616252660751,
      0.06216215342283249,
      0.07939725369215012,
      0.0003976755542680621,
      0.09540800005197525,
      0.09262751787900925,
      0.09039735049009323,
      0.08823440223932266,
      0.08617508411407471,
      0.08421764522790909,
      0.09382311999797821,
      0.1093239039182663,
      0.02097838930785656,
      0.09758554399013519,
      0.12619110941886902,
      0.1064719632267952,
      0.07429461926221848,
      0.09576396644115448,
      0.12894080579280853,
      0.11987552046775818,
      0.12903062999248505,
      0.11312293261289597,
      0.060313500463962555,
      0.05634469538927078,
      0.02792077139019966,
      0.11698418110609055,
      0.06294634193181992,
      0.10262063145637512,
      0.07690133899450302,
      0.06856132298707962,
      0.06587245315313339,
      0.09937164187431335,
      0.09740942716598511,
      0.11809708178043365,
      0.04260704666376114,
      0.018327387049794197,
      0.08196623623371124,
      0.0976773276925087,
      0.05842491611838341,
      0.05452478304505348,
      0.05261804163455963,
      0.050821613520383835,
      0.049069274216890335,
      0.04738852009177208,
      0.03425709530711174,
      0.02564571239054203,
      0.05088833346962929,
      0.00026594827068038285,
      0.057784684002399445,
      0.09850101172924042,
      0.03000403754413128,
      0.10293938964605331,
      0.11998935788869858,
      0.00041006284300237894,
      0.030806496739387512,
      0.033233676105737686,
      0.015799762681126595,
      0.02577856369316578,
      0.06506489962339401,
      0.05957970395684242,
      0.055434636771678925,
      0.0770367830991745,
      0.09105204045772552,
      0.00035427825059741735,
      0.00018998155428562313,
      0.10930465906858444,
      0.003743458539247513,
      0.01961556449532509,
      0.0006788435275666416,
      0.01586838811635971,
      0.007490916643291712,
      0.0004246703756507486,
      0.0009721893584355712,
      0.001906099496409297,
      0.0002029447932727635,
      0.19015198945999146,
      0.01128893531858921,
      0.15740671753883362,
      4.73568450079509e-13
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " =",
      " sum",
      "(",
      "sc",
      "ores",
      ")",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " total",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " [",
      "0",
      " for",
      " _",
      " in",
      " scores",
      "]",
      "\n",
      " ",
      " ",
      " ",
      " normalized",
      " =",
      " []",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " s",
      " in",
      " scores",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " normalized",
      ".",
      "append",
      "(",
      "s",
      " /",
      " total",
      ")",
      "\n",
      " ",
      " ",
      " ",
      " return",
      " normalized",
      "\n",
      "\n",
      "def",
      " __",
      "init",
      "__",
      " (",
      " self",
      " ,",
      " score",
      " ):",
      "\n",
      "\n",
      "self",
      " .",
      "score",
      " =",
      "\n",
      "\n",
      "def",
      " __"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      60,
      61,
      62,
      63,
      64,
      65,
      66,
      67,
      68,
      69,
      70,
      71,
      72,
      73,
      74,
      75,
      76,
      77,
      78,
      79,
      80,
      81,
      82,
      83,
      84,
      85,
      86,
      87,
      88,
      89,
      90,
      91,
      92,
      93,
      94,
      95,
      96,
      97,
      98,
      99,
      100,
      102,
      103,
      104,
      105
    ],
    "token": "str"
  },
  "107": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.014001294039189816,
      0.7760366797447205,
      6.694685339425632e-07
    ],
    "rationales": [
      "__",
      " score",
      "str"
    ],
    "rationales_indexes": [
      89,
      93,
      106
    ],
    "token": "__"
  },
  "108": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.1752055138349533,
      0.03551758453249931,
      1.1657400591502665e-06
    ],
    "rationales": [
      ")",
      " ):",
      "__"
    ],
    "rationales_indexes": [
      77,
      94,
      107
    ],
    "token": " ("
  },
  "109": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      9.32038456085138e-05,
      0.029435550794005394,
      0.0007233729120343924,
      0.07781586796045303,
      0.21182988584041595,
      0.13134627044200897,
      0.26249873638153076,
      1.7202773960889317e-06
    ],
    "rationales": [
      "def",
      " ):",
      "self",
      " .",
      " =",
      "\n",
      "def",
      " ("
    ],
    "rationales_indexes": [
      0,
      94,
      97,
      98,
      100,
      103,
      104,
      108
    ],
    "token": " self"
  },
  "11": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      2.3805437194823753e-06,
      8.995742064143997e-06,
      2.834840870491462e-06,
      3.190868483216036e-06,
      0.0002190017985412851,
      4.19317120758933e-06,
      6.895029946463183e-05,
      0.00031939722248353064,
      1.2942364264745265e-05,
      2.3464535843231715e-05,
      1.6497244459401372e-08
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10
    ],
    "token": " "
  },
  "110": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.005647911690175533,
      0.10535237193107605,
      0.05084233358502388,
      0.08504261076450348,
      0.04296258091926575,
      0.06545569747686386,
      0.03389116749167442,
      0.11789751797914505,
      0.023580966517329216,
      0.11416914314031601,
      0.11712886393070221,
      0.09976273775100708,
      0.011353355832397938,
      0.11942622810602188,
      0.13500110805034637,
      0.13142456114292145,
      0.12662118673324585,
      0.12163270264863968,
      0.03164656460285187,
      0.09332600980997086,
      1.057261211911964e-08
    ],
    "rationales": [
      "def",
      "ize",
      "_",
      "sc",
      "(",
      "\n",
      " ",
      " ",
      "\n",
      "\n",
      " ",
      " ",
      "(",
      " /",
      "\n",
      " ",
      " ",
      " ",
      " ,",
      "\n",
      " self"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      4,
      6,
      10,
      11,
      12,
      21,
      30,
      47,
      48,
      73,
      75,
      78,
      79,
      80,
      81,
      92,
      103,
      109
    ],
    "token": " ,"
  },
  "111": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.002383922692388296,
      0.0003986639203503728,
      0.0007389640086330473,
      0.0004863042267970741,
      0.0005854738992638886,
      0.000513415434397757,
      0.0006295480416156352,
      0.00038377600139938295,
      0.0008292141719721258,
      0.00038338371086865664,
      0.0004182574921287596,
      0.000480581569718197,
      0.0008512603235431015,
      0.0006897817365825176,
      0.0007668475736863911,
      0.0005744625232182443,
      0.0007211352931335568,
      0.0003686188720166683,
      0.0008734292932786047,
      0.0007673724903725088,
      0.00033705434179864824,
      0.00040984107181429863,
      0.0006511611863970757,
      0.0005327895050868392,
      0.0007339384756051004,
      0.0007279954734258354,
      0.000861833686940372,
      0.0008361994987353683,
      0.0007096889312379062,
      0.0004342634929344058,
      0.0006696552154608071,
      0.0009441778529435396,
      0.0008870525052770972,
      0.0004545986885204911,
      0.0008040046668611467,
      0.0004768538346979767,
      0.00043496518628671765,
      0.0008106568711809814,
      0.0005957720568403602,
      0.0009332027984783053,
      0.0008590556681156158,
      0.0007804613560438156,
      0.0008600200526416302,
      0.0005297461175359786,
      0.0004453880537766963,
      0.0008984878659248352,
      0.00042998595745302737,
      0.0006771592306904495,
      0.0007494591409340501,
      0.0006198043702170253,
      0.0005617516580969095,
      0.06842702627182007,
      0.023373886942863464,
      0.05799737572669983,
      0.0003744100104086101,
      0.07528097927570343,
      0.04594239592552185,
      0.004937273915857077,
      0.011934744194149971,
      0.0033916879910975695,
      7.457505125785246e-05,
      0.0009596981108188629,
      6.305836450337665e-06
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "(",
      "sc",
      "\n",
      " total",
      " =",
      "(",
      "sc",
      ")",
      "\n",
      " ",
      " if",
      " total",
      ":",
      "\n",
      " ",
      " return",
      " [",
      "0",
      " _",
      "]",
      "\n",
      " ",
      " ",
      " ",
      " normalized",
      " []",
      " ",
      " ",
      " ",
      " s",
      " scores",
      ":",
      " ",
      " ",
      " ",
      " ",
      ".",
      "append",
      "(",
      "s",
      " ",
      " return",
      " __",
      "init",
      "__",
      " ,",
      " score",
      " ):",
      "self",
      " .",
      " =",
      "\n",
      "def",
      " __",
      "str",
      " (",
      " self",
      " ,"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      6,
      7,
      10,
      14,
      15,
      17,
      18,
      20,
      21,
      22,
      25,
      26,
      29,
      30,
      32,
      38,
      39,
      40,
      42,
      45,
      46,
      47,
      48,
      49,
      50,
      52,
      54,
      55,
      56,
      58,
      60,
      61,
      63,
      64,
      65,
      66,
      71,
      72,
      73,
      74,
      79,
      82,
      87,
      88,
      89,
      92,
      93,
      94,
      97,
      98,
      100,
      102,
      104,
      105,
      106,
      108,
      109,
      110
    ],
    "token": " name"
  },
  "112": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.3968285620212555,
      0.0049178991466760635,
      8.685890861670487e-06
    ],
    "rationales": [
      "def",
      " ,",
      " name"
    ],
    "rationales_indexes": [
      0,
      110,
      111
    ],
    "token": " ,"
  },
  "113": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      4.058798003825359e-05,
      0.004093968775123358,
      0.0022829733788967133,
      0.009310266003012657,
      0.013261922635138035,
      0.007741089444607496,
      0.00828168448060751,
      0.005944726523011923,
      0.0010744535829871893,
      0.011684540659189224,
      0.013200928457081318,
      0.007054124493151903,
      0.0065729375928640366,
      0.013003728352487087,
      0.013714033178985119,
      0.012024858966469765,
      0.012573780491948128,
      0.006423879414796829,
      0.01132606714963913,
      0.011306727305054665,
      0.01314176432788372,
      0.00632926169782877,
      0.010539672337472439,
      0.013315433636307716,
      0.012681998312473297,
      0.012157551012933254,
      0.012287121266126633,
      0.012371589429676533,
      0.012436924502253532,
      0.005143897142261267,
      0.004662771243602037,
      0.009146882221102715,
      0.009520883671939373,
      0.010078842751681805,
      0.009656243026256561,
      0.005783890839666128,
      0.006751795765012503,
      0.010547677055001259,
      0.014212451875209808,
      0.011466270312666893,
      0.010670337826013565,
      0.01347870659083128,
      0.01404961571097374,
      0.010811690241098404,
      0.07197482883930206,
      0.009294159710407257,
      0.058072496205568314,
      0.0033499186392873526,
      0.04626844823360443,
      0.01362038403749466,
      0.03841666504740715,
      0.020178934559226036,
      0.010317613370716572,
      5.567881544266129e-06
    ],
    "rationales": [
      "def",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      " total",
      " =",
      " sum",
      "sc",
      "ores",
      ")",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " total",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " [",
      "0",
      " ",
      " ",
      " ",
      " normalized",
      " =",
      " ",
      " ",
      " ",
      " for",
      ":",
      " ",
      " ",
      " normalized",
      " self",
      " score",
      " score",
      "def",
      " (",
      " self",
      " ,",
      " name",
      " ,"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      4,
      5,
      6,
      14,
      15,
      16,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      47,
      48,
      49,
      50,
      51,
      54,
      55,
      56,
      57,
      61,
      63,
      68,
      83,
      91,
      93,
      101,
      104,
      108,
      109,
      110,
      111,
      112
    ],
    "token": " value"
  },
  "114": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.31607678532600403,
      0.015611864626407623,
      1.828416541435618e-12
    ],
    "rationales": [
      " return",
      " ):",
      " value"
    ],
    "rationales_indexes": [
      38,
      94,
      113
    ],
    "token": " ):"
  },
  "115": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.606633722782135,
      0.006717168726027012
    ],
    "rationales": [
      "\n",
      " ):"
    ],
    "rationales_indexes": [
      103,
      114
    ],
    "token": "\n"
  },
  "116": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.9966083765029907
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      115
    ],
    "token": "\n"
  },
  "117": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.0001463975349906832,
      0.01845838874578476,
      0.05809501186013222,
      0.039166636765003204,
      0.10387071222066879,
      0.13368746638298035,
      0.078145332634449,
      0.0027363079134374857,
      7.62989497971045e-12
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "]",
      " __",
      "str",
      " ):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      45,
      105,
      106,
      114,
      116
    ],
    "token": "\"\"\""
  },
  "118": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.7311646938323975,
      0.0010938337072730064
    ],
    "rationales": [
      "\n",
      "\"\"\""
    ],
    "rationales_indexes": [
      116,
      117
    ],
    "token": "\n"
  },
  "119": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.9958972930908203
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      118
    ],
    "token": "\n"
  },
  "12": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.9994699358940125
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      11
    ],
    "token": " "
  },
  "13": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9994625449180603
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      12
    ],
    "token": " "
  },
  "14": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      1.4032398212293629e-05,
      9.45210485951975e-05,
      5.584535392699763e-05,
      7.737764099147171e-05,
      8.174942195182666e-05,
      2.300852611369919e-05,
      1.105838509829482e-05,
      0.00036315087345428765,
      0.00011110895866295323,
      7.95599989942275e-05,
      5.2674127800855786e-05,
      7.46975711081177e-05,
      0.00043465805356390774,
      9.373120946065683e-09
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13
    ],
    "token": " total"
  },
  "15": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.005307035055011511,
      0.11839868873357773,
      0.3145912289619446,
      0.2397080957889557,
      0.16555888950824738,
      0.019862046465277672,
      0.2077489197254181,
      2.6081521387055773e-09
    ],
    "rationales": [
      "def",
      " normal",
      "sc",
      "(",
      "ores",
      "):",
      " ",
      " total"
    ],
    "rationales_indexes": [
      0,
      1,
      4,
      6,
      8,
      9,
      11,
      14
    ],
    "token": " ="
  },
  "16": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.009920799173414707,
      0.033143043518066406,
      0.038913875818252563,
      0.03935522958636284,
      0.02963864989578724,
      0.013621365651488304,
      0.026178428903222084,
      0.030038364231586456,
      0.0044690510258078575,
      0.004641749896109104,
      0.004692221526056528,
      0.005614650901407003,
      0.002573410514742136,
      0.004776529502123594,
      0.0006377037498168647,
      1.4821001741438522e-06
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " ="
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15
    ],
    "token": " sum"
  },
  "17": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.005268946290016174,
      0.41019365191459656,
      0.02629266120493412,
      0.05958978459239006,
      3.5601350756309103e-09
    ],
    "rationales": [
      "def",
      " normal",
      "):",
      " =",
      " sum"
    ],
    "rationales_indexes": [
      0,
      1,
      9,
      15,
      16
    ],
    "token": "("
  },
  "18": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Focal Method"
    ],
    "probabilities": [
      0.00034435911220498383,
      0.0036769125144928694,
      0.0010443973587825894,
      0.04167509078979492,
      5.038661754497298e-08
    ],
    "rationales": [
      "def",
      "_",
      "sc",
      "):",
      "("
    ],
    "rationales_indexes": [
      0,
      3,
      4,
      9,
      17
    ],
    "token": "sc"
  },
  "19": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.3501524031162262,
      0.14319506287574768,
      2.658660491761111e-07
    ],
    "rationales": [
      "ores",
      "ores",
      "sc"
    ],
    "rationales_indexes": [
      5,
      8,
      18
    ],
    "token": "ores"
  },
  "2": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.16358184814453125,
      1.3186226555106373e-09
    ],
    "rationales": [
      "def",
      " normal"
    ],
    "rationales_indexes": [
      0,
      1
    ],
    "token": "ize"
  },
  "20": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.008684920147061348,
      0.1653469353914261,
      0.11794683337211609,
      2.3001271074463148e-06
    ],
    "rationales": [
      "def",
      " sum",
      "(",
      "ores"
    ],
    "rationales_indexes": [
      0,
      16,
      17,
      19
    ],
    "token": ")"
  },
  "21": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.09801681339740753
    ],
    "rationales": [
      ")"
    ],
    "rationales_indexes": [
      20
    ],
    "token": "\n"
  },
  "22": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.7646499872207642,
      0.10179124027490616,
      0.0009831251809373498,
      1.2532242976703856e-07
    ],
    "rationales": [
      " ",
      " ",
      " total",
      "\n"
    ],
    "rationales_indexes": [
      12,
      13,
      14,
      21
    ],
    "token": " "
  },
  "23": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.999372661113739
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      22
    ],
    "token": " "
  },
  "24": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Conditional"
    ],
    "probabilities": [
      0.9994456171989441
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      23
    ],
    "token": " "
  },
  "25": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.005904674530029297,
      0.03922131657600403,
      0.008666420355439186,
      0.03287292644381523,
      0.04205121845006943,
      0.04478738084435463,
      0.0005939124966971576,
      0.010715373791754246,
      0.026211369782686234,
      0.02611520327627659,
      0.018513645976781845,
      0.005562446545809507,
      0.0039906492456793785,
      0.019863322377204895,
      0.0123459342867136,
      0.010859792120754719,
      0.02398226410150528,
      0.03864031657576561,
      0.014886828139424324,
      0.008333667181432247,
      0.01901005581021309,
      0.005388508550822735,
      0.018646936863660812,
      0.023174190893769264,
      1.0255162408157048e-07
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " =",
      " sum",
      "(",
      "sc",
      "ores",
      ")",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24
    ],
    "token": " if"
  },
  "26": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.039253003895282745,
      0.12508128583431244,
      0.004972035996615887,
      0.0220194011926651,
      0.17749956250190735,
      0.002457014750689268,
      0.06908103823661804,
      0.01772015541791916,
      0.0001714426907710731,
      0.0035071279853582382,
      0.026666611433029175,
      1.294424123443605e-06
    ],
    "rationales": [
      "def",
      "ize",
      "(",
      "sc",
      "):",
      " total",
      " =",
      " sum",
      "(",
      "\n",
      " ",
      " if"
    ],
    "rationales_indexes": [
      0,
      2,
      6,
      7,
      9,
      14,
      15,
      16,
      17,
      21,
      24,
      25
    ],
    "token": " total"
  },
  "27": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      4.938096026307903e-05,
      0.19096888601779938,
      0.07287930697202682,
      0.0007553435862064362,
      0.027280066162347794,
      0.009617790579795837,
      0.0026223016902804375,
      6.107440508190676e-12
    ],
    "rationales": [
      "def",
      "):",
      " total",
      " =",
      "ores",
      ")",
      " if",
      " total"
    ],
    "rationales_indexes": [
      0,
      9,
      14,
      15,
      19,
      20,
      25,
      26
    ],
    "token": " =="
  },
  "28": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.03343593701720238,
      0.012741206213831902,
      0.09289923310279846,
      0.014532228000462055,
      0.057832397520542145,
      0.3814507722854614,
      0.1111736074090004,
      5.473167220770847e-06
    ],
    "rationales": [
      "sc",
      "\n",
      " total",
      ")",
      "\n",
      " if",
      " total",
      " =="
    ],
    "rationales_indexes": [
      4,
      10,
      14,
      20,
      21,
      25,
      26,
      27
    ],
    "token": " 0"
  },
  "29": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.0020262044854462147,
      0.010741522535681725,
      0.03450404107570648,
      0.004287193529307842,
      0.0030923299491405487,
      0.02752322144806385,
      0.19429276883602142,
      0.061506692320108414,
      0.0002216221473645419
    ],
    "rationales": [
      "def",
      "):",
      " total",
      " =",
      "sc",
      ")",
      " ",
      " if",
      " 0"
    ],
    "rationales_indexes": [
      0,
      9,
      14,
      15,
      18,
      20,
      24,
      25,
      28
    ],
    "token": ":"
  },
  "3": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.07390201836824417,
      0.043185725808143616,
      1.3090255379211158e-05
    ],
    "rationales": [
      "def",
      " normal",
      "ize"
    ],
    "rationales_indexes": [
      0,
      1,
      2
    ],
    "token": "_"
  },
  "30": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.2474105805158615,
      0.04836852103471756,
      0.6614924073219299,
      4.491244453674881e-06
    ],
    "rationales": [
      "sc",
      ")",
      "\n",
      ":"
    ],
    "rationales_indexes": [
      18,
      20,
      21,
      29
    ],
    "token": "\n"
  },
  "31": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.8593998551368713,
      0.0010300484718754888,
      0.058331795036792755,
      2.9933983114460716e-07
    ],
    "rationales": [
      " ",
      " ",
      " total",
      "\n"
    ],
    "rationales_indexes": [
      23,
      24,
      26,
      30
    ],
    "token": " "
  },
  "32": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.9992952346801758
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      31
    ],
    "token": " "
  },
  "33": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.9994258880615234
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      32
    ],
    "token": " "
  },
  "34": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9992619156837463
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      33
    ],
    "token": " "
  },
  "35": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9993482232093811
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      34
    ],
    "token": " "
  },
  "36": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9992576241493225
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      35
    ],
    "token": " "
  },
  "37": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      0.9992762207984924
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      36
    ],
    "token": " "
  },
  "38": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Bool"
    ],
    "probabilities": [
      2.1566927898675203e-05,
      0.011265048757195473,
      0.0586986280977726,
      0.02400755137205124,
      0.03868203982710838,
      0.003452480770647526,
      0.00045179453445598483,
      5.417889470393789e-10
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "):",
      " if",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      9,
      25,
      37
    ],
    "token": " return"
  },
  "39": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.004309311043471098,
      0.03942754864692688,
      0.08456549048423767,
      0.07637228071689606,
      0.02319953590631485,
      0.06618788838386536,
      0.08990389853715897,
      6.747701490894542e-07
    ],
    "rationales": [
      "def",
      "_",
      "ores",
      "ores",
      "):",
      "ores",
      " 0",
      " return"
    ],
    "rationales_indexes": [
      0,
      3,
      5,
      8,
      9,
      19,
      28,
      38
    ],
    "token": " ["
  },
  "4": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.0009135193540714681,
      0.001378348795697093,
      0.0017028546426445246,
      1.4846276030766603e-07
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3
    ],
    "token": "sc"
  },
  "40": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Punctuation"
    ],
    "probabilities": [
      0.007047116756439209,
      0.07172413170337677,
      0.027585627511143684,
      0.01835581474006176,
      0.006292854435741901,
      0.06319579482078552,
      0.05427372083067894,
      0.0032152505591511726,
      0.03979397565126419,
      4.869973781751469e-05
    ],
    "rationales": [
      "def",
      "_",
      "ores",
      "(",
      " total",
      "sc",
      " total",
      " 0",
      ":",
      " ["
    ],
    "rationales_indexes": [
      0,
      3,
      5,
      6,
      14,
      18,
      26,
      28,
      29,
      39
    ],
    "token": "0"
  },
  "41": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Signature"
    ],
    "probabilities": [
      0.0036156943533569574,
      0.00553593784570694,
      0.004723768215626478,
      0.07573583722114563,
      0.04808430373668671,
      0.009481986053287983,
      0.034629471600055695,
      0.07242905348539352,
      0.008408172987401485,
      0.060190267860889435,
      0.03496316447854042,
      0.05661223083734512,
      0.05324419587850571,
      0.06408306956291199,
      0.041525352746248245,
      0.046067699790000916,
      0.008081600069999695,
      0.04040459170937538,
      0.06772837787866592,
      0.004350605886429548,
      0.029782256111502647,
      0.008973738178610802,
      0.06032751128077507,
      0.10722333937883377,
      0.10797979682683945,
      0.0007606534636579454,
      0.029270995408296585,
      0.06519819051027298,
      0.045564234256744385,
      0.009968776255846024,
      0.007937383837997913,
      0.10743948072195053,
      0.08424817770719528,
      0.06420405954122543,
      0.06663541495800018,
      0.06960228830575943,
      0.0747622549533844,
      0.09789256751537323,
      0.1097622737288475,
      0.026341566815972328,
      1.7813284785006545e-06
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " =",
      " sum",
      "(",
      "sc",
      "ores",
      ")",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " total",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " [",
      "0"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40
    ],
    "token": " for"
  },
  "42": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.0005718239699490368,
      0.1056620329618454,
      0.0028159725479781628,
      0.123581163585186,
      0.05135403200984001,
      0.1265803724527359,
      0.15071342885494232,
      0.16083376109600067,
      0.15693393349647522,
      0.010569829493761063,
      0.15514543652534485,
      0.12714159488677979,
      0.16781029105186462,
      0.17592214047908783,
      0.12636321783065796,
      0.12188173830509186,
      0.1203460544347763,
      0.11562896519899368,
      0.09070084244012833,
      0.07411101460456848,
      0.17955735325813293,
      0.14396129548549652,
      5.978294836950226e-08
    ],
    "rationales": [
      "def",
      " normal",
      "_",
      "sc",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " =",
      ")",
      "\n",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      "0",
      " for"
    ],
    "rationales_indexes": [
      0,
      1,
      3,
      4,
      9,
      10,
      11,
      12,
      13,
      15,
      20,
      21,
      29,
      30,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      40,
      41
    ],
    "token": " _"
  },
  "43": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adverb"
    ],
    "probabilities": [
      0.058124277740716934,
      0.10349725186824799,
      0.00411397498100996,
      6.309389988246039e-08
    ],
    "rationales": [
      " [",
      "0",
      " for",
      " _"
    ],
    "rationales_indexes": [
      39,
      40,
      41,
      42
    ],
    "token": " in"
  },
  "44": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      2.2963852188695455e-06,
      0.05282343924045563,
      0.05540245771408081,
      0.011694608256220818,
      0.0001577975635882467,
      2.1827496311743744e-05,
      0.035925619304180145,
      0.013863754458725452,
      0.0009000989375635982,
      0.018018366768956184,
      0.0588308684527874,
      0.002833704464137554,
      0.01390735525637865,
      0.011198061518371105,
      0.07526487112045288,
      0.0759800523519516,
      0.03475752845406532,
      0.05986209213733673,
      0.004538880195468664,
      0.0012957071885466576,
      0.007877947762608528,
      0.011451289057731628,
      0.011767827905714512,
      0.013343256898224354,
      0.014311742037534714,
      0.04899849370121956,
      0.011443465016782284,
      0.011739143170416355,
      0.05687430873513222,
      0.01319920364767313,
      0.012176265940070152,
      0.01421349123120308,
      0.012596487998962402,
      0.07186733186244965,
      0.01345762424170971,
      0.01198296993970871,
      0.011329560540616512,
      0.00973923783749342,
      0.0763167068362236,
      0.04949937015771866,
      0.006050115916877985,
      0.037157606333494186,
      0.005210870411247015,
      9.879314255556437e-09
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " =",
      " sum",
      "(",
      "sc",
      "ores",
      ")",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " total",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " [",
      "0",
      " for",
      " _",
      " in"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43
    ],
    "token": " scores"
  },
  "45": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.25626057386398315,
      1.2035512142460902e-08
    ],
    "rationales": [
      " [",
      " scores"
    ],
    "rationales_indexes": [
      39,
      44
    ],
    "token": "]"
  },
  "46": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.31543534994125366,
      0.00209296983666718
    ],
    "rationales": [
      "def",
      "]"
    ],
    "rationales_indexes": [
      0,
      45
    ],
    "token": "\n"
  },
  "47": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.029332634061574936,
      0.015114105306565762,
      0.16701941192150116,
      0.3533429801464081,
      0.0074503375217318535,
      0.03080575540661812,
      0.010645803064107895,
      0.05024239420890808,
      0.009884980507194996,
      0.01388612948358059,
      0.2534089684486389,
      0.009143426083028316,
      0.04240573197603226,
      0.5115119218826294,
      0.02900911681354046,
      0.006542866118252277,
      0.056072648614645004,
      0.0018112323014065623,
      1.5548742737792054e-07
    ],
    "rationales": [
      " normal",
      "ores",
      "\n",
      " =",
      " sum",
      "sc",
      "ores",
      ")",
      " ",
      " ",
      " ",
      " if",
      " 0",
      "\n",
      " ",
      " for",
      " scores",
      "]",
      "\n"
    ],
    "rationales_indexes": [
      1,
      5,
      10,
      15,
      16,
      18,
      19,
      20,
      22,
      23,
      24,
      25,
      28,
      30,
      31,
      41,
      44,
      45,
      46
    ],
    "token": " "
  },
  "48": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.999299168586731
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      47
    ],
    "token": " "
  },
  "49": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9995001554489136
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      48
    ],
    "token": " "
  },
  "5": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.002030447358265519,
      0.03936004638671875,
      0.03808392584323883,
      0.018115095794200897,
      2.40642208382269e-07
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4
    ],
    "token": "ores"
  },
  "50": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      3.109863939698698e-07,
      4.061244453623658e-06,
      1.2719009646389168e-05,
      0.0009453757083974779,
      3.907636346411891e-05,
      0.0009094597189687192,
      0.0008824956021271646,
      0.00022428618103731424,
      0.0013516241451725364,
      2.3360043996945024e-05,
      0.0016660216497257352,
      0.0002569324860814959,
      0.0008538338588550687,
      0.0007141990936361253,
      0.00037862834869883955,
      0.0015217493055388331,
      0.0010536893969401717,
      0.0014408438000828028,
      0.0009532668045721948,
      0.0008506356971338391,
      0.00011673345579765737,
      0.0007753281970508397,
      0.0009726411080919206,
      0.0007324530743062496,
      0.0016339949797838926,
      0.00013426251825876534,
      0.0012893268140032887,
      0.0011508525349199772,
      0.00014233175897970796,
      0.0015022390289232135,
      0.0012200275668874383,
      0.0006710152374580503,
      0.00018326996359974146,
      0.000469664839329198,
      0.0003147119132336229,
      0.00035016293986700475,
      0.0003980912151746452,
      6.625540117966011e-05,
      1.376370278194372e-06,
      1.8456576071912423e-05,
      0.0007311111548915505,
      0.000272947596386075,
      0.0007286628824658692,
      0.0005646620411425829,
      0.0011753654107451439,
      0.0007137736538425088,
      0.00018217696924693882,
      0.0007787185604684055,
      0.0009720613597892225,
      1.1459001878887792e-13
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " =",
      " sum",
      "(",
      "sc",
      "ores",
      ")",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " total",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " [",
      "0",
      " for",
      " _",
      " in",
      " scores",
      "]",
      "\n",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49
    ],
    "token": " normalized"
  },
  "51": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.0047857738099992275,
      0.5633984208106995,
      0.11013013869524002,
      1.458088760841747e-08
    ],
    "rationales": [
      "def",
      " total",
      " =",
      " normalized"
    ],
    "rationales_indexes": [
      0,
      14,
      15,
      50
    ],
    "token": " ="
  },
  "52": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Operators"
    ],
    "probabilities": [
      0.0004726850602310151,
      0.0461590550839901,
      0.02426200918853283,
      0.008879777044057846,
      0.027269259095191956,
      0.019262978807091713,
      0.04925939068198204,
      0.031876612454652786,
      0.011768966913223267,
      0.006278716027736664,
      0.04287727549672127,
      0.05492965131998062,
      0.04058027267456055,
      0.038042765110731125,
      0.018808167427778244,
      0.035482242703437805,
      0.05025327578186989,
      0.03959400951862335,
      0.015290994197130203,
      0.030534138903021812,
      0.018334783613681793,
      0.05286237224936485,
      0.01763855665922165,
      0.03840544447302818,
      0.029706427827477455,
      0.05188749358057976,
      0.018123922869563103,
      0.05348854511976242,
      0.03842853009700775,
      0.016868602484464645,
      0.053578827530145645,
      0.056174203753471375,
      0.05582199990749359,
      0.04793776944279671,
      0.04479749873280525,
      0.0399637334048748,
      0.03781851753592491,
      0.03133213892579079,
      0.03094855695962906,
      0.00020578208204824477,
      0.0003003785677719861,
      0.03519884869456291,
      0.023105068132281303,
      0.032539110630750656,
      0.00026366361998952925,
      0.0026545762084424496,
      0.007916421629488468,
      0.01723269745707512,
      0.011332531459629536,
      0.01402390468865633,
      0.005245935171842575,
      1.5315491452838614e-07
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " =",
      " sum",
      "(",
      "sc",
      "ores",
      ")",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " total",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " [",
      "0",
      " for",
      " _",
      " in",
      " scores",
      "]",
      "\n",
      " ",
      " ",
      " ",
      " normalized",
      " ="
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51
    ],
    "token": " []"
  },
  "53": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.712379515171051,
      5.409733816463813e-08
    ],
    "rationales": [
      "\n",
      " []"
    ],
    "rationales_indexes": [
      46,
      52
    ],
    "token": "\n"
  },
  "54": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.7039836645126343,
      0.376126766204834,
      0.13629695773124695,
      0.017864473164081573,
      6.543690744820196e-08
    ],
    "rationales": [
      " ",
      " ",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      36,
      47,
      48,
      49,
      53
    ],
    "token": " "
  },
  "55": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Return"
    ],
    "probabilities": [
      0.9993746876716614
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      54
    ],
    "token": " "
  },
  "56": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.9994377493858337
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      55
    ],
    "token": " "
  },
  "57": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.009154772385954857,
      0.17682509124279022,
      0.14371532201766968,
      0.11896680295467377,
      0.09196723997592926,
      0.055250950157642365,
      0.0013315076939761639,
      0.0009626272949390113,
      0.0006902753957547247,
      0.0005482124397531152,
      0.0012667158152908087,
      0.0012896149419248104,
      0.0007686045137234032,
      0.0012884847819805145,
      0.0006516518187709153,
      0.003203148953616619,
      0.0014929514145478606,
      0.2015364170074463,
      0.027158204466104507,
      0.2110193371772766,
      0.01650022156536579,
      0.30014219880104065,
      0.10832738876342773,
      0.05210135504603386,
      0.016765844076871872,
      0.02025287039577961,
      0.19780747592449188,
      0.017752211540937424,
      9.290356928204346e-08
    ],
    "rationales": [
      "def",
      "_",
      "ores",
      "):",
      " total",
      " =",
      " sum",
      "(",
      "sc",
      "\n",
      " ",
      " ",
      " ",
      " if",
      "\n",
      " return",
      " for",
      " _",
      " in",
      " scores",
      "\n",
      " ",
      " ",
      " ",
      " =",
      " []",
      " ",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      3,
      8,
      9,
      14,
      15,
      16,
      17,
      18,
      21,
      22,
      23,
      24,
      25,
      30,
      38,
      41,
      42,
      43,
      44,
      46,
      47,
      48,
      49,
      51,
      52,
      54,
      55,
      56
    ],
    "token": " for"
  },
  "58": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Comment"
    ],
    "probabilities": [
      0.0020868766587227583,
      0.006748972460627556,
      0.019887646660208702,
      0.02635188214480877,
      0.020456552505493164,
      0.003314883913844824,
      0.02306765876710415,
      0.004944188054651022,
      0.015623919665813446,
      0.02344437688589096,
      0.022702962160110474,
      0.016313860192894936,
      0.016569441184401512,
      0.01522815227508545,
      0.013783684000372887,
      0.016405748203396797,
      0.007949860766530037,
      0.01785830780863762,
      0.01302122138440609,
      0.016209285706281662,
      0.021649274975061417,
      0.016718145459890366,
      0.016442006453871727,
      0.026050185784697533,
      0.016306666657328606,
      0.026516294106841087,
      0.008616411127150059,
      0.011712932959198952,
      0.010863170959055424,
      0.010923845693469048,
      0.016335781663656235,
      0.029620572924613953,
      0.025560900568962097,
      0.02608843706548214,
      0.024662107229232788,
      0.02460576966404915,
      0.024736100807785988,
      0.024807177484035492,
      0.024610329419374466,
      0.02882322482764721,
      0.02604050189256668,
      0.007639170624315739,
      0.008984746411442757,
      0.009436392225325108,
      0.003989868331700563,
      0.023693149909377098,
      0.025915754958987236,
      0.016246169805526733,
      0.01641133427619934,
      0.02951643615961075,
      0.026358148083090782,
      0.018054192885756493,
      0.028311975300312042,
      0.016233691945672035,
      0.025264672935009003,
      0.016575047746300697,
      0.016364116221666336,
      6.866014882689342e-05
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " =",
      " sum",
      "(",
      "sc",
      "ores",
      ")",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " total",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " [",
      "0",
      " for",
      " _",
      " in",
      " scores",
      "]",
      "\n",
      " ",
      " ",
      " ",
      " normalized",
      " =",
      " []",
      "\n",
      " ",
      " ",
      " ",
      " for"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57
    ],
    "token": " s"
  },
  "59": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.23506447672843933,
      0.004044890869408846,
      0.012778696604073048,
      0.10945446044206619,
      0.1299292892217636,
      0.09237146377563477,
      0.00020750823023263365
    ],
    "rationales": [
      "ores",
      " total",
      " in",
      " scores",
      " normalized",
      " =",
      " s"
    ],
    "rationales_indexes": [
      19,
      26,
      43,
      44,
      50,
      51,
      58
    ],
    "token": " in"
  },
  "6": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.018816770985722542,
      0.26819556951522827,
      0.12103603780269623,
      6.371507765834394e-07
    ],
    "rationales": [
      "def",
      "ize",
      "_",
      "ores"
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      5
    ],
    "token": "("
  },
  "60": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      8.914812497096136e-05,
      0.0007634102366864681,
      0.0002864101843442768,
      0.009887156076729298,
      0.047881852835416794,
      0.0023970985785126686,
      0.0041080075316131115,
      0.0350906178355217,
      0.05937252938747406,
      0.10328318923711777,
      0.06770450621843338,
      0.09695730358362198,
      0.11684595048427582,
      0.02439695969223976,
      0.25767213106155396,
      7.856616321078036e-06,
      0.05409638583660126,
      0.016072839498519897,
      0.08124297112226486,
      7.53261986119469e-09
    ],
    "rationales": [
      "def",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      "ores",
      " 0",
      " ",
      " return",
      "0",
      " _",
      " in",
      " scores",
      " ",
      "\n",
      " s",
      " in"
    ],
    "rationales_indexes": [
      0,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      19,
      28,
      31,
      38,
      40,
      42,
      43,
      44,
      49,
      53,
      58,
      59
    ],
    "token": " scores"
  },
  "61": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.06966555118560791,
      0.17630623281002045,
      0.04223983734846115,
      0.1018955409526825,
      0.10025445371866226,
      0.12990297377109528,
      0.03755883499979973,
      0.03482668474316597,
      2.1527544902255613e-07
    ],
    "rationales": [
      " sum",
      "sc",
      " total",
      " 0",
      ":",
      " scores",
      " =",
      "\n",
      " scores"
    ],
    "rationales_indexes": [
      16,
      18,
      26,
      28,
      29,
      44,
      51,
      53,
      60
    ],
    "token": ":"
  },
  "62": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.6365391612052917,
      0.27939844131469727,
      1.949942770806956e-06
    ],
    "rationales": [
      "\n",
      " normalized",
      ":"
    ],
    "rationales_indexes": [
      46,
      50,
      61
    ],
    "token": "\n"
  },
  "63": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.5146631002426147,
      0.23039154708385468,
      0.06567616015672684,
      0.011959012597799301,
      2.1949430717427276e-08
    ],
    "rationales": [
      " ",
      " ",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      48,
      54,
      55,
      56,
      62
    ],
    "token": " "
  },
  "64": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.9994564652442932
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      63
    ],
    "token": " "
  },
  "65": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Cardinal"
    ],
    "probabilities": [
      0.9994423985481262
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      64
    ],
    "token": " "
  },
  "66": {
    "concept_view": [
      "Programming Language",
      "Semantic",
      "Loops"
    ],
    "probabilities": [
      0.999432384967804
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      65
    ],
    "token": " "
  },
  "67": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Adjective"
    ],
    "probabilities": [
      0.9995028972625732
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      66
    ],
    "token": " "
  },
  "68": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "List"
    ],
    "probabilities": [
      0.9993764758110046
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      67
    ],
    "token": " "
  },
  "69": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.9994613528251648
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      68
    ],
    "token": " "
  },
  "7": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00036754130269400775,
      0.02393842674791813,
      0.00468922033905983,
      0.0009314029011875391,
      0.01388580072671175,
      4.199687708705824e-08
    ],
    "rationales": [
      "def",
      "ize",
      "_",
      "sc",
      "ores",
      "("
    ],
    "rationales_indexes": [
      0,
      2,
      3,
      4,
      5,
      6
    ],
    "token": "sc"
  },
  "70": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Conjunction"
    ],
    "probabilities": [
      3.655327134310937e-07,
      4.0398426790488884e-05,
      0.00014726887457072735,
      0.04002872481942177,
      9.665677498560399e-05,
      0.0008593560196459293,
      0.00028464835486374795,
      0.0026417917106300592,
      0.025467492640018463,
      0.036937251687049866,
      0.0346359945833683,
      0.006119075696915388,
      0.004178033210337162,
      0.030620547011494637,
      1.0725241736508906e-05,
      0.004748101811856031,
      0.0065933638252317905,
      0.003488915041089058,
      2.941108959930716e-06,
      0.019603824242949486,
      0.0016646513249725103,
      2.123818194377236e-05,
      0.053944095969200134,
      0.009212542325258255,
      1.3568782565359563e-14
    ],
    "rationales": [
      "def",
      " normal",
      "_",
      "\n",
      " ",
      " ",
      " total",
      " =",
      " sum",
      "ores",
      " total",
      ":",
      "\n",
      " ",
      " return",
      " [",
      "\n",
      " ",
      " normalized",
      " =",
      "\n",
      " scores",
      "\n",
      " ",
      " "
    ],
    "rationales_indexes": [
      0,
      1,
      3,
      10,
      11,
      13,
      14,
      15,
      16,
      19,
      26,
      29,
      30,
      32,
      38,
      39,
      46,
      49,
      50,
      51,
      53,
      60,
      62,
      63,
      69
    ],
    "token": " normalized"
  },
  "71": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.07151637971401215,
      0.03883522376418114,
      0.03391923010349274,
      0.22348570823669434,
      0.03629602491855621,
      0.029402485117316246,
      0.03966405615210533,
      0.04119763895869255,
      0.055250126868486404,
      0.0428309291601181,
      0.040073007345199585,
      0.03289259225130081,
      0.0306407380849123,
      0.029787903651595116,
      0.028697585687041283,
      0.030596978962421417,
      0.029008809477090836,
      0.029826829209923744,
      0.029491746798157692,
      0.029345350340008736,
      0.028457142412662506,
      0.03839851915836334,
      0.027854224666953087,
      0.04270239546895027,
      0.03932866454124451,
      0.029341477900743484,
      0.04259437695145607,
      0.27172335982322693,
      0.027998775243759155,
      0.04024948179721832,
      0.04180028662085533,
      0.0421285480260849,
      0.03749869018793106,
      0.08477652072906494,
      0.030538253486156464,
      0.03012404777109623,
      0.1542530059814453,
      0.03487425297498703,
      0.02889455482363701,
      0.029936745762825012,
      0.029856713488698006,
      0.028018560260534286,
      9.450328661841922e-07
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "ores",
      "sc",
      "ores",
      "sc",
      "ores",
      ")",
      " ",
      " if",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " [",
      " for",
      " _",
      " in",
      " scores",
      "]",
      "\n",
      " ",
      " normalized",
      " =",
      " []",
      " ",
      " for",
      " s",
      " in",
      ":",
      "\n",
      " ",
      " ",
      " normalized"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      5,
      7,
      8,
      18,
      19,
      20,
      22,
      25,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      39,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      50,
      51,
      52,
      54,
      57,
      58,
      59,
      61,
      62,
      63,
      64,
      70
    ],
    "token": "."
  },
  "72": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      8.950452752287674e-07,
      0.0026913543697446585,
      0.0005665492499247193,
      0.0015479503199458122,
      0.004787691403180361,
      0.010095593519508839,
      0.007578473538160324,
      0.007001285906881094,
      0.00904866959899664,
      1.880802847153973e-05,
      0.0033506613690406084,
      0.00600967975333333,
      0.008557090535759926,
      0.004335396457463503,
      0.005067881662398577,
      0.055566731840372086,
      0.0410209521651268,
      0.021188845857977867,
      0.012773790396749973,
      1.6794382551310605e-10
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      " sum",
      "sc",
      "ores",
      ")",
      "\n",
      " if",
      " in",
      " scores",
      "\n",
      "."
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      16,
      18,
      19,
      20,
      21,
      25,
      59,
      60,
      62,
      71
    ],
    "token": "append"
  },
  "73": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      0.06054937094449997,
      0.20524033904075623,
      0.12699609994888306,
      1.6529116919627995e-07
    ],
    "rationales": [
      "def",
      " normal",
      "):",
      "append"
    ],
    "rationales_indexes": [
      0,
      1,
      9,
      72
    ],
    "token": "("
  },
  "74": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      0.025369158014655113,
      0.00613806489855051,
      0.05306696146726608,
      0.025183863937854767,
      0.03381337970495224,
      0.002944606589153409,
      0.03415518254041672,
      0.054419200867414474,
      0.05995125323534012,
      0.05651969462633133,
      0.028621573001146317,
      0.032489534467458725,
      0.11477410048246384,
      0.179138645529747,
      0.07958914339542389,
      0.005024754907935858,
      0.013886232860386372,
      0.03539346531033516,
      0.010297446511685848,
      0.0054246000945568085,
      0.01175658032298088,
      0.045824985951185226,
      0.03352868929505348,
      0.02253904566168785,
      0.036195769906044006,
      7.627917511854321e-05
    ],
    "rationales": [
      "def",
      "sc",
      "(",
      "sc",
      " sum",
      ")",
      " ",
      " total",
      " ",
      " ",
      " ",
      " in",
      " =",
      " ",
      " ",
      " for",
      " s",
      " in",
      " scores",
      ":",
      "\n",
      " ",
      " ",
      ".",
      "append",
      "("
    ],
    "rationales_indexes": [
      0,
      4,
      6,
      7,
      16,
      20,
      22,
      26,
      31,
      32,
      33,
      43,
      51,
      55,
      56,
      57,
      58,
      59,
      60,
      61,
      62,
      63,
      69,
      71,
      72,
      73
    ],
    "token": "s"
  },
  "75": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.0011175749823451042,
      0.029896123334765434,
      0.030721111223101616,
      0.03463658690452576,
      0.031192956492304802,
      0.033163540065288544,
      0.007488371804356575,
      0.027202896773815155,
      0.029612606391310692,
      0.001659519039094448,
      0.001222517341375351,
      0.03270866721868515,
      0.030481494963169098,
      0.003692168742418289,
      0.007127442397177219,
      0.00574445491656661,
      0.0027369349263608456,
      0.006666812114417553,
      0.0012118960730731487,
      0.020579909905791283,
      0.0058544171042740345,
      0.02366422861814499,
      0.026615703478455544,
      0.02684619277715683,
      0.026316246017813683,
      0.023642363026738167,
      0.02251303941011429,
      0.02327546291053295,
      0.01588265970349312,
      0.01041184738278389,
      0.01991329900920391,
      0.019285524263978004,
      0.01888524368405342,
      0.01852993853390217,
      0.018137041479349136,
      0.017740918323397636,
      0.01742156594991684,
      0.01716291718184948,
      0.02559858374297619,
      0.016450533643364906,
      0.02651788853108883,
      0.020361637696623802,
      0.018142832443118095,
      0.012924754060804844,
      0.006712416652590036,
      0.023106979206204414,
      0.011152956634759903,
      0.025979062542319298,
      0.0255315899848938,
      0.02478433959186077,
      0.02571040578186512,
      0.008905579335987568,
      0.001682330621406436,
      0.026919957250356674,
      0.02233874425292015,
      0.021602969616651535,
      0.021020369604229927,
      0.02642437443137169,
      0.014891880564391613,
      0.0034066291991621256,
      0.020519813522696495,
      0.024266695603728294,
      0.03570710867643356,
      0.02413085661828518,
      0.0212860107421875,
      0.020492179319262505,
      0.015133701264858246,
      0.014059143140912056,
      0.01183379627764225,
      0.021792205050587654,
      0.03341754525899887,
      0.019657744094729424,
      0.0004366091452538967,
      0.007463615853339434,
      4.113970135222189e-05
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " =",
      " sum",
      "(",
      "sc",
      "ores",
      ")",
      "\n",
      " ",
      " ",
      " ",
      " if",
      " total",
      " ==",
      " 0",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      " [",
      "0",
      " for",
      " _",
      " in",
      " scores",
      "]",
      "\n",
      " ",
      " ",
      " ",
      " normalized",
      " =",
      " []",
      "\n",
      " ",
      " ",
      " ",
      " for",
      " s",
      " in",
      " scores",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " normalized",
      ".",
      "append",
      "(",
      "s"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      60,
      61,
      62,
      63,
      64,
      65,
      66,
      67,
      68,
      69,
      70,
      71,
      72,
      73,
      74
    ],
    "token": " /"
  },
  "76": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.061457276344299316,
      0.026941141113638878,
      0.0757962167263031,
      0.1041698157787323,
      0.08284246176481247,
      0.0704813152551651,
      0.07315832376480103,
      0.10865219682455063,
      0.07578184455633163,
      0.05499182641506195,
      0.0009103224729187787,
      0.06512459367513657,
      0.07377097755670547,
      0.07456348091363907,
      0.08190498501062393,
      0.1007738783955574,
      0.0038851944264024496,
      0.08823905885219574,
      0.05817528814077377,
      0.07417290657758713,
      0.06587594002485275,
      0.05527956411242485,
      0.07500891387462616,
      0.07030975073575974,
      0.0970631018280983,
      0.09528798609972,
      0.09136490523815155,
      0.0005217728321440518,
      0.0509752593934536,
      0.0671854168176651,
      0.09085249900817871,
      0.09617964178323746,
      0.015371344983577728,
      0.0700727254152298,
      0.08218997716903687,
      0.09249083697795868,
      0.09053831547498703,
      0.10678529739379883,
      0.04420049488544464,
      0.04759691283106804,
      0.08725636452436447,
      0.047574590891599655,
      0.058568403124809265,
      0.08283961564302444,
      0.061423905193805695,
      0.053662944585084915,
      0.047009821981191635,
      0.0358261838555336,
      0.0028828163631260395,
      0.00020191962539684027,
      0.0006635679746977985,
      2.4201913220167626e-06
    ],
    "rationales": [
      " normal",
      "ize",
      "_",
      "sc",
      "sc",
      "\n",
      " ",
      " ",
      " ",
      " sum",
      "(",
      "sc",
      "\n",
      " ",
      " ",
      " ",
      " total",
      " ==",
      ":",
      "\n",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " ",
      " return",
      "0",
      " for",
      " _",
      " in",
      " scores",
      "\n",
      " ",
      " ",
      " normalized",
      " =",
      " []",
      "\n",
      " ",
      " ",
      " s",
      " scores",
      ":",
      " ",
      " ",
      " normalized",
      "append",
      "(",
      "s",
      " /"
    ],
    "rationales_indexes": [
      1,
      2,
      3,
      4,
      7,
      10,
      11,
      12,
      13,
      16,
      17,
      18,
      21,
      22,
      23,
      24,
      26,
      27,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      40,
      41,
      42,
      43,
      44,
      46,
      47,
      48,
      50,
      51,
      52,
      53,
      54,
      55,
      58,
      60,
      61,
      65,
      66,
      70,
      72,
      73,
      74,
      75
    ],
    "token": " total"
  },
  "77": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Particle"
    ],
    "probabilities": [
      0.17791710793972015,
      0.010304841212928295,
      7.743302177232181e-08
    ],
    "rationales": [
      "append",
      "(",
      " total"
    ],
    "rationales_indexes": [
      72,
      73,
      76
    ],
    "token": ")"
  },
  "78": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.5300542712211609,
      0.039649881422519684
    ],
    "rationales": [
      " if",
      ")"
    ],
    "rationales_indexes": [
      25,
      77
    ],
    "token": "\n"
  },
  "79": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Determiner"
    ],
    "probabilities": [
      0.6805334687232971,
      0.39526697993278503,
      0.08483024686574936,
      0.2509957551956177,
      0.020082006230950356,
      5.754638987554017e-09
    ],
    "rationales": [
      " ",
      " ",
      " ",
      " ",
      " ",
      "\n"
    ],
    "rationales_indexes": [
      48,
      64,
      67,
      68,
      69,
      78
    ],
    "token": " "
  },
  "8": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.9707453846931458,
      3.065572968807828e-07
    ],
    "rationales": [
      "ores",
      "sc"
    ],
    "rationales_indexes": [
      5,
      7
    ],
    "token": "ores"
  },
  "80": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9995658993721008
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      79
    ],
    "token": " "
  },
  "81": {
    "concept_view": [
      "Programming Language",
      "Syntax",
      "Errors"
    ],
    "probabilities": [
      0.9995823502540588
    ],
    "rationales": [
      " "
    ],
    "rationales_indexes": [
      80
    ],
    "token": " "
  },
  "82": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Class"
    ],
    "probabilities": [
      3.7477420846698806e-05,
      0.07088164985179901,
      0.01577545329928398,
      0.002324762288480997,
      0.08741381019353867,
      0.05743028596043587,
      0.0318199098110199,
      0.007740590255707502,
      1.6810186576066144e-10
    ],
    "rationales": [
      "def",
      " if",
      " total",
      " return",
      "0",
      " =",
      " for",
      " scores",
      " "
    ],
    "rationales_indexes": [
      0,
      25,
      26,
      38,
      40,
      51,
      57,
      60,
      81
    ],
    "token": " return"
  },
  "83": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      3.200756736987387e-06,
      7.51867555663921e-05,
      0.0004985675914213061,
      0.01615981012582779,
      0.0037892849650233984,
      0.019106484949588776,
      0.048587579280138016,
      0.04051097109913826,
      0.009823573753237724,
      0.03603450208902359,
      0.024644948542118073,
      0.09414147585630417,
      0.06306108087301254,
      0.07819976657629013,
      0.041048504412174225,
      3.103146628546938e-10
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      " =",
      "\n",
      " if",
      " ==",
      " normalized",
      " normalized",
      ".",
      "s",
      " /",
      " total",
      ")",
      " return"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      15,
      21,
      25,
      27,
      50,
      70,
      71,
      74,
      75,
      76,
      77,
      82
    ],
    "token": " normalized"
  },
  "84": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.9159671664237976,
      1.387245356454514e-05
    ],
    "rationales": [
      "\n",
      " normalized"
    ],
    "rationales_indexes": [
      78,
      83
    ],
    "token": "\n"
  },
  "85": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Constructor"
    ],
    "probabilities": [
      0.9995298385620117
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      84
    ],
    "token": "\n"
  },
  "86": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.00045330874854698777,
      0.03858814761042595,
      0.007931183092296124,
      0.12883581221103668,
      0.0032855377066880465,
      0.17300915718078613,
      0.07295376807451248,
      5.515293517027731e-12
    ],
    "rationales": [
      "def",
      "_",
      "):",
      "\n",
      " return",
      "\n",
      " return",
      "\n"
    ],
    "rationales_indexes": [
      0,
      3,
      9,
      10,
      38,
      53,
      82,
      85
    ],
    "token": "def"
  },
  "87": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      0.001017913338728249,
      0.028056545183062553,
      0.032281745225191116,
      0.011497629806399345,
      0.10672920942306519,
      0.04165070131421089,
      0.05297524854540825,
      0.09883707761764526,
      0.11221020668745041,
      0.0049776905216276646,
      0.020650405436754227,
      0.02433180995285511,
      0.0900535061955452,
      0.0763997808098793,
      0.06377536803483963,
      0.1029762327671051,
      0.059130147099494934,
      0.08331438899040222,
      0.0378536619246006,
      6.999196244805717e-08
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores",
      "):",
      " ",
      " ",
      " ",
      ")",
      " ",
      " ",
      " if",
      " ",
      "\n",
      "def"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      11,
      12,
      13,
      20,
      23,
      24,
      25,
      31,
      62,
      86
    ],
    "token": " __"
  },
  "88": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.00019056453311350197,
      0.004658335819840431,
      0.02418554201722145,
      0.011227541603147984,
      0.00020428949210327119,
      6.01065949012991e-06,
      0.0007158872904255986,
      2.7056945225978346e-11
    ],
    "rationales": [
      "def",
      "append",
      " return",
      " normalized",
      "\n",
      "\n",
      "def",
      " __"
    ],
    "rationales_indexes": [
      0,
      72,
      82,
      83,
      84,
      85,
      86,
      87
    ],
    "token": "init"
  },
  "89": {
    "concept_view": [
      "Programming Language",
      "Context Window",
      "Field"
    ],
    "probabilities": [
      0.4396618902683258,
      0.015167217701673508,
      2.7011606493210216e-11
    ],
    "rationales": [
      " scores",
      " __",
      "init"
    ],
    "rationales_indexes": [
      60,
      87,
      88
    ],
    "token": "__"
  },
  "9": {
    "concept_view": [
      "Programming Language",
      "Non-Semantic",
      "Expression"
    ],
    "probabilities": [
      0.0016460210317745805,
      0.016263596713542938,
      0.01078311912715435,
      0.005194955505430698,
      0.01741155982017517,
      0.013463923707604408,
      0.0018952277023345232,
      0.018450720235705376,
      4.8864684742966347e-08
    ],
    "rationales": [
      "def",
      " normal",
      "ize",
      "_",
      "sc",
      "ores",
      "(",
      "sc",
      "ores"
    ],
    "rationales_indexes": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8
    ],
    "token": "):"
  },
  "90": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.07740660011768341,
      0.10906015336513519,
      1.994328158616554e-06
    ],
    "rationales": [
      ")",
      " __",
      "__"
    ],
    "rationales_indexes": [
      77,
      87,
      89
    ],
    "token": " ("
  },
  "91": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.005809825845062733,
      0.052455052733421326,
      0.003885646816343069,
      0.025122100487351418,
      0.02874264493584633,
      0.023527342826128006,
      0.00552999647334218,
      0.012491795234382153,
      0.7669939398765564,
      0.004080152604728937,
      0.004439373500645161,
      0.014543358236551285,
      0.010662413202226162,
      0.006284513045102358,
      0.0016649519093334675,
      0.00020487559959292412,
      0.0287692341953516,
      0.010294183157384396,
      2.0286124708945863e-06
    ],
    "rationales": [
      "def",
      "(",
      " for",
      " in",
      "append",
      "s",
      " /",
      " total",
      ")",
      "\n",
      " ",
      " return",
      " normalized",
      "\n",
      "\n",
      "def",
      " __",
      "init",
      " ("
    ],
    "rationales_indexes": [
      0,
      6,
      57,
      59,
      72,
      74,
      75,
      76,
      77,
      78,
      81,
      82,
      83,
      84,
      85,
      86,
      87,
      88,
      90
    ],
    "token": " self"
  },
  "92": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Preposition"
    ],
    "probabilities": [
      0.005211330484598875,
      0.42380771040916443,
      0.31153056025505066,
      0.10379670560359955,
      0.009014574810862541,
      0.15322405099868774,
      0.08199512958526611,
      0.07195544987916946,
      0.12323808670043945,
      0.023511270061135292,
      1.1523760612419665e-08
    ],
    "rationales": [
      "def",
      "append",
      " return",
      "\n",
      "\n",
      "def",
      " __",
      "init",
      "__",
      " (",
      " self"
    ],
    "rationales_indexes": [
      0,
      72,
      82,
      84,
      85,
      86,
      87,
      88,
      89,
      90,
      91
    ],
    "token": " ,"
  },
  "93": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Verb"
    ],
    "probabilities": [
      3.559240440154099e-06,
      0.059301454573869705,
      0.060859911143779755,
      0.003689092816784978,
      0.007029524073004723,
      0.011180942878127098,
      0.014776407741010189,
      0.02631749026477337,
      0.00010910322453128174,
      0.06638303399085999,
      0.00025756630930118263,
      0.00013736714026890695,
      1.721313310554251e-05,
      3.787415835176944e-06,
      0.00012392211647238582,
      0.018758509308099747,
      0.033903755247592926,
      0.0002908421738538891,
      0.040498021990060806,
      0.00026703436742536724,
      3.2301006740453886e-06,
      0.05448225885629654,
      1.0352250683354214e-05,
      0.0493738055229187,
      0.0008819230715744197,
      0.022528093308210373,
      0.09409081935882568,
      4.691027061198838e-05,
      2.7731286536436528e-05,
      0.08962028473615646,
      0.08006779849529266,
      6.2898379837861285e-06,
      0.04552610218524933,
      0.09619051963090897,
      0.10860389471054077,
      0.002128711435943842,
      0.10514478385448456,
      0.00019749536295421422,
      0.07295189052820206,
      0.12916657328605652,
      7.764581937408366e-07
    ],
    "rationales": [
      "def",
      "_",
      "ores",
      "(",
      "sc",
      "ores",
      " total",
      "(",
      " if",
      " total",
      " ==",
      " 0",
      " ",
      " ",
      " return",
      "0",
      " _",
      " scores",
      "\n",
      " normalized",
      " =",
      " ",
      " ",
      " ",
      " scores",
      ":",
      " ",
      " ",
      " ",
      " ",
      " ",
      " normalized",
      ".",
      "append",
      " total",
      ")",
      " ",
      " return",
      " normalized",
      " self",
      " ,"
    ],
    "rationales_indexes": [
      0,
      3,
      5,
      6,
      7,
      8,
      14,
      17,
      25,
      26,
      27,
      28,
      31,
      32,
      38,
      40,
      42,
      44,
      46,
      50,
      51,
      54,
      55,
      56,
      60,
      61,
      65,
      66,
      67,
      68,
      69,
      70,
      71,
      72,
      76,
      77,
      81,
      82,
      83,
      91,
      92
    ],
    "token": " score"
  },
  "94": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "String"
    ],
    "probabilities": [
      6.250480510061607e-06,
      0.156429722905159,
      0.11362878233194351,
      0.006866624113172293,
      0.002629769267514348,
      0.0005451684701256454,
      0.018222922459244728,
      0.04350252449512482,
      0.027887826785445213,
      0.026297694072127342,
      0.02441377565264702,
      0.025782465934753418,
      0.02758665569126606,
      0.019201451912522316,
      0.01674424670636654,
      0.02603844366967678,
      0.01940508931875229,
      0.029075389727950096,
      0.0107530876994133,
      0.06600794196128845,
      0.0255373977124691,
      0.39452800154685974,
      0.32567861676216125,
      0.2598981261253357,
      0.004385441541671753,
      0.02766726352274418,
      0.030924824997782707,
      0.1949140876531601,
      0.0002675988944247365,
      0.0009628236293792725,
      0.0090163080021739,
      2.69380416410836e-12
    ],
    "rationales": [
      "def",
      " normal",
      "sc",
      "ores",
      "ores",
      "):",
      " ",
      " ",
      " sum",
      "ores",
      " ",
      "\n",
      "\n",
      " for",
      " s",
      " in",
      ":",
      "append",
      "(",
      " total",
      ")",
      " ",
      " ",
      " ",
      " return",
      "\n",
      "def",
      "__",
      " (",
      " self",
      " ,",
      " score"
    ],
    "rationales_indexes": [
      0,
      1,
      4,
      5,
      8,
      9,
      11,
      12,
      16,
      19,
      24,
      30,
      46,
      57,
      58,
      59,
      61,
      72,
      73,
      76,
      77,
      79,
      80,
      81,
      82,
      84,
      86,
      89,
      90,
      91,
      92,
      93
    ],
    "token": " ):"
  },
  "95": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "Noun"
    ],
    "probabilities": [
      0.7376409769058228,
      0.006196279078722
    ],
    "rationales": [
      " /",
      " ):"
    ],
    "rationales_indexes": [
      75,
      94
    ],
    "token": "\n"
  },
  "96": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.9994078874588013
    ],
    "rationales": [
      "\n"
    ],
    "rationales_indexes": [
      95
    ],
    "token": "\n"
  },
  "97": {
    "concept_view": [
      "Programming Language",
      "Natural Language in Code",
      "Identifier"
    ],
    "probabilities": [
      0.0009025656618177891,
      0.0008823135867714882,
      0.10472637414932251,
      0.029576405882835388,
      0.0001992995967157185,
      3.723830133139927e-08
    ],
    "rationales": [
      "def",
      " total",
      " __",
      " self",
      " ):",
      "\n"
    ],
    "rationales_indexes": [
      0,
      26,
      87,
      91,
      94,
      96
    ],
    "token": "self"
  },
  "98": {
    "concept_view": [
      "Natural Language",
      "Semantic",
      "pronouns"
    ],
    "probabilities": [
      0.006951753981411457,
      0.12074296176433563,
      0.5944284796714783,
      8.815241592241563e-11
    ],
    "rationales": [
      "def",
      " ):",
      "\n",
      "self"
    ],
    "rationales_indexes": [
      0,
      94,
      96,
      97
    ],
    "token": " ."
  },
  "99": {
    "concept_view": [
      "Natural Language",
      "Non-semantic",
      "Model"
    ],
    "probabilities": [
      0.0015821048291400075,
      0.006905184593051672,
      3.467829856163007e-06,
      0.019207710400223732,
      0.0037758529651910067,
      1.7192472796523361e-06,
      8.876886568032205e-05,
      0.0001402625784976408,
      7.4243611258850706e-09
    ],
    "rationales": [
      "def",
      "init",
      " self",
      " ,",
      " score",
      " ):",
      "\n",
      "self",
      " ."
    ],
    "rationales_indexes": [
      0,
      88,
      91,
      92,
      93,
      94,
      96,
      97,
      98
    ],
    "token": "score"
  },
  "_phrase": "def normalize_scores(scores):\n    total = sum(scores)\n    if total == 0:\n        return [0 for _ in scores]\n    normalized = []\n    for s in scores:\n        normalized.append(s / total)\n    return normalized\n\ndef __init__ ( self , score ):\n\nself .score = score\n\ndef __str__ ( self , name , value ):\n\n\"\"\"\n\n"
}
curl -X POST http://127.0.0.1:5000/prompt -H "Content-Type: application/json"  0.03s user 0.03s system 0% cpu 9:48.19 total

