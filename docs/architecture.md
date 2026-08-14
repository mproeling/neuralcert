# Architectuur

## Generieke laag

`neuracert.core` definieert alleen contracts en value objects. Een problem
plugin implementeert domeinsampling, een differentiable objective en goedkope
validatie. De algoritmen leven in afzonderlijke stages:

```text
Problem
  │
  ▼
Discovery ──> Distillation ──> Refinement ──> Verification
    │              │               │                │
 numeriek       structureel      lokaal/exact     onafhankelijk
```

De pipeline bewaart iedere tussenuitkomst als een typed stage result. Geen
stage mag een numeriek resultaat stilzwijgend promoveren tot een bewijs.

`neuracert.problems.maynard` is een plugin/adaptatielaag. De gespecialiseerde
historische implementatie blijft voorlopig onder `maynard_tools`, zodat de
nieuwe abstractie geen bewezen werkende numeriek herschrijft zonder afzonderlijke
regressievalidatie.

## Harde grens

`maynard_tools.discovery` mag geen module uit
`maynard_tools.certification` importeren. Deze regel wordt met een AST-test
gecontroleerd en omvat ook dynamische imports waarvan de module als letterlijke
string is opgegeven.

De reden is inhoudelijk: discovery rekent snel en numeriek om kandidaten te
vinden; certification moet zelfstandig vaststellen wat exact bewezen is.

## Dependencyrichting

```text
maynard_tools.discovery ──> NumPy / SciPy / PyTorch
maynard_tools.certification ──> NumPy / SciPy / optioneel FLINT
```

Certification mag NPZ-uitvoer van discovery als **data** lezen. Dat is geen
Python-import en creëert geen code-afhankelijkheid in de andere richting.

## Historische varianten

De twee discovery-entrypoints blijven afzonderlijk omdat de gated variant een
eigen distributed uitvoeringsmodel en uitgebreidere objective-logica heeft.
Alleen aantoonbaar identieke definities zijn naar gedeelde modules verplaatst.

De certificeringsvarianten blijven expliciete backends. Zo kan een bestaand
resultaat met dezelfde methode opnieuw worden gecontroleerd zonder stilzwijgend
naar een andere rekenmethode over te schakelen.

De publieke discovery- en certification-CLI's gebruiken dezelfde methodenamen:
`poly` voor de bestaande polynomial/neural workflow en `ratio` voor de
confluent-ratio workflow. De ratio-certifier gebruikt Arb met directed rounding
en accepteert momenteel alleen één power-1-kanaal.

## Onafhankelijke verifier

`maynard_tools.verifier` is alleen voor distributiegemak onderdeel van hetzelfde
wheel. De implementatie importeert niets uit `discovery` of `certification` en
heeft geen relatieve imports naar andere packagecode. Een AST-test bewaakt beide
regels. De certificaatverifier gebruikt alleen de standaardbibliotheek voor de
poly-route en laadt NumPy en python-flint pas binnen de ratio-route. De directe
I/J-controle gebruikt NumPy en is expliciet niet-rigoureus.
