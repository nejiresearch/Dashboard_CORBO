//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
// ExportRietveld_HSPlus                                                //
//                                                                      //
// Exporte, pour le document HighScore Plus actif (anchor dataset) :    //
//  1) <scan>_profil.csv  : un point par ligne                          //
//     2theta ; d (A) ; Q (A-1) ; I_obs ; I_calc ; I_bkg ; I_diff ;      //
//     sigma ; w_resid ; I_<phase 1> ; I_<phase 2> ; ...                 //
//  2) <scan>_rietveld.log : caracteristiques du fit Rietveld           //
//                                                                      //
// A lancer APRES l'affinement Rietveld (manuellement ou en batch).     //
// Fichiers ecrits dans GetUserFolder (Documents\PANalytical\HighScore) //
//                                                                      //
// ORGANISATION DU SCRIPT                                               //
//  - Section A : appels CONFIRMES (tires du script d'exemple PANalytical)//
//  - Section B : appels A VERIFIER (noms probables, non confirmes).    //
//    Chaque appel incertain est isole dans une petite fonction.        //
//    Si HighScore signale "Unknown identifier" sur une ligne de la     //
//    section B : chercher le bon nom avec Ctrl+Espace dans l'editeur,  //
//    ou remplacer le corps de la fonction par  Result := NA;           //
//    (la valeur sera alors ecrite comme -999 et le reste fonctionnera).//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

program ExportRietveldHSPlus;

const
  // ---- Parametres utilisateur ------------------------------------------
  WAVELENGTH        = 1.540598;  // Longueur d'onde K-alpha1 en Angstrom (Cu)
                             // Co : 1.788965 ; Mo : 0.709300 ; Fe : 1.936042
  EXPORT_POINTS = True;      // False = ne produire que le .log (test rapide)
  SEP           = ';';       // separateur CSV (';' compatible Excel FR)
  NA            = -999.0;    // valeur ecrite si une grandeur est indisponible

var
  Csv, Log: TStringList;
  BaseName: string;

//======================================================================//
// Utilitaires                                                          //
//======================================================================//

function F(Value: Double; NoOfDigits: Integer): string;
begin
  Result := Format('%1.' + IntToStr(NoOfDigits) + 'f', [Value]);
end;

function CleanName(S: string): string;
// remplace les caracteres genants pour un nom de colonne / fichier
begin
  Result := StringReplace(S,      ' ', '_', [rfReplaceAll]);
  Result := StringReplace(Result, ';', '_', [rfReplaceAll]);
  Result := StringReplace(Result, '\', '_', [rfReplaceAll]);
  Result := StringReplace(Result, '/', '_', [rfReplaceAll]);
  Result := StringReplace(Result, ':', '_', [rfReplaceAll]);
  Result := StringReplace(Result, '.', '_', [rfReplaceAll]);
end;

procedure CheckIfDocExists;
begin
  if not Assigned(XDoc) then
  begin
    ShowMessage('Aucun document HighScore ouvert : script arrete.');
    Halt;
  end;
end;

//======================================================================//
// SECTION A : appels CONFIRMES (script d'exemple PANalytical v4.1)     //
//======================================================================//

function ScanFileName: string;
begin
  Result := XDoc.AnchorDataSet.Scan.ScanProperties.FileName;
end;

function ScanShortName: string;
begin
  Result := XDoc.AnchorDataSet.Scan.ScanProperties.Name;
end;

function ScanTimeStamp: string;
begin
  Result := XDoc.AnchorDataSet.Scan.ScanProperties.StartTimeStamp;
end;

function SampleID: string;
begin
  Result := XDoc.AnchorDataSet.Scan.ScanProperties.SampleID;
end;

function PhaseCount: Integer;
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.Count;
end;

function PhaseTitle(p: Integer): string;
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].Title;
end;

function PhaseUsed(p: Integer): Boolean;
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].UsePhase;
end;

function PhaseWeightPct(p: Integer): Double;
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.WeightPercentage(p, ScanKAlpha1Wave).Value;
end;

function RProfile: Double;   // Rp
begin
  Result := XDoc.AnchorDataSet.Structure.GlobalParameters.RProfile;
end;

function GOF: Double;
begin
  Result := XDoc.AnchorDataSet.Structure.GlobalParameters.GoodnessOfFit;
end;

//======================================================================//
// SECTION B : appels A VERIFIER (noms probables)                       //
// -> en cas d'erreur de compilation, corriger ici ou mettre Result:=NA //
//======================================================================//

// ---- Indices d'accord globaux ----------------------------------------
function RWeightedProfile: Double;   // Rwp
begin
  Result := XDoc.AnchorDataSet.Structure.GlobalParameters.RWeightedProfile;
end;

function RExpected: Double;          // Rexp
begin
  Result := XDoc.AnchorDataSet.Structure.GlobalParameters.RExpected;
end;

function ZeroShift: Double;          // decalage du zero (deg 2theta)
begin
  Result := XDoc.AnchorDataSet.Structure.GlobalParameters.ZeroShift.Value;
end;

function SpecimenDisplacement: Double; // deplacement echantillon (mm)
begin
  Result := XDoc.AnchorDataSet.Structure.GlobalParameters.SpecimenDisplacement.Value;
end;

// ---- Parametres par phase --------------------------------------------
function PhaseWeightESD(p: Integer): Double;  // incertitude sur le % massique
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.WeightPercentage(p, ScanKAlpha1Wave).ESD;
end;

function PhaseScale(p: Integer): Double;
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].ScaleFactor.Value;
end;

function PhaseRBragg(p: Integer): Double;
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].RBragg;
end;

function PhaseSpaceGroup(p: Integer): string;
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].SpaceGroup.Name;
end;

function PhaseCell(p, k: Integer): Double;  // k = 0..5 : a b c alpha beta gamma
begin
  case k of
    0: Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].A.Value;
    1: Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].B.Value;
    2: Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].C.Value;
    3: Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].Alpha.Value;
    4: Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].Beta.Value;
    5: Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].Gamma.Value;
  else
    Result := NA;
  end;
end;

function PhaseVolume(p: Integer): Double;
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].CellVolume;
end;

function PhaseProfileUVW(p, k: Integer): Double; // k = 0..2 : U V W
begin
  case k of
    0: Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].U.Value;
    1: Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].V.Value;
    2: Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].W.Value;
  else
    Result := NA;
  end;
end;

// ---- Donnees point par point -----------------------------------------
function NbPoints: Integer;
begin
  Result := XDoc.AnchorDataSet.Scan.Count;
end;

function Pos2Theta(i: Integer): Double;
begin
  Result := XDoc.AnchorDataSet.Scan.Positions[i];
end;

function IObs(i: Integer): Double;
begin
  Result := XDoc.AnchorDataSet.Scan.Intensities[i];
end;

function ICalc(i: Integer): Double;
begin
  Result := XDoc.AnchorDataSet.CalculatedProfile.Intensities[i];
end;

function IBkg(i: Integer): Double;
begin
  Result := XDoc.AnchorDataSet.Background.Intensities[i];
end;

function IPhase(p, i: Integer): Double;  // profil calcule de la phase p
begin
  Result := XDoc.AnchorDataSet.Structure.Phases.Items[p].CalculatedProfile.Intensities[i];
end;

//======================================================================//
// Ecriture du profil point par point                                   //
//======================================================================//

procedure WriteProfile;
var
  i, p, n, np: Integer;
  line: string;
  vTth, vTh, vObs, vCalc, vSig: Double;
begin
  np := PhaseCount;
  line := '2theta_deg' + SEP + 'd_A' + SEP + 'Q_A-1' + SEP + 'I_obs' + SEP +
          'I_calc' + SEP + 'I_bkg' + SEP + 'I_diff' + SEP + 'sigma' + SEP + 'w_resid';
  for p := 0 to np - 1 do
    line := line + SEP + 'I_' + CleanName(PhaseTitle(p));
  Csv.Add(line);

  n := NbPoints;
  for i := 0 to n - 1 do
  begin
    vTth  := Pos2Theta(i);
    vTh   := vTth / 2.0 * Pi / 180.0;
    vObs  := IObs(i);
    vCalc := ICalc(i);
    if vObs > 0 then vSig := Sqrt(vObs) else vSig := 1.0;

    line := F(vTth, 4) + SEP;
    if Sin(vTh) > 0 then
      line := line + F(WAVELENGTH / (2.0 * Sin(vTh)), 5) + SEP +
                     F(4.0 * Pi * Sin(vTh) / WAVELENGTH, 5) + SEP
    else
      line := line + F(NA, 0) + SEP + F(NA, 0) + SEP;
    line := line + F(vObs, 2) + SEP + F(vCalc, 2) + SEP + F(IBkg(i), 2) + SEP +
                   F(vObs - vCalc, 2) + SEP + F(vSig, 3) + SEP +
                   F((vObs - vCalc) / vSig, 4);
    for p := 0 to np - 1 do
      line := line + SEP + F(IPhase(p, i), 2);
    Csv.Add(line);
  end;
end;

//======================================================================//
// Ecriture du .log                                                     //
//======================================================================//

procedure WriteLog;
var
  p, k: Integer;
  total: Double;
const
  CellNames: array[0..5] of string = ('a (A)            ', 'b (A)            ', 'c (A)            ',
                                      'alpha (deg)      ', 'beta (deg)       ', 'gamma (deg)      ');
  UVWNames: array[0..2] of string = ('U', 'V', 'W');
begin
  Log.Add('==================================================================');
  Log.Add(' RAPPORT D''AFFINEMENT RIETVELD - HighScore Plus');
  Log.Add('==================================================================');
  Log.Add('Date export          : ' + DateTimeToStr(Now));
  Log.Add('Fichier scan         : ' + ScanFileName);
  Log.Add('Date mesure          : ' + ScanTimeStamp);
  Log.Add('Sample ID            : ' + SampleID);
  Log.Add('Longueur d''onde (A)  : ' + F(WAVELENGTH, 6) + '  (constante du script)');
  if EXPORT_POINTS then
  begin
    Log.Add('Nombre de points     : ' + IntToStr(NbPoints));
    Log.Add('Plage 2theta (deg)   : ' + F(Pos2Theta(0), 4) + ' - ' + F(Pos2Theta(NbPoints - 1), 4));
    if NbPoints > 1 then
      Log.Add('Pas (deg)            : ' + F(Pos2Theta(1) - Pos2Theta(0), 5));
  end;
  Log.Add('');
  Log.Add('------------------------- INDICES D''ACCORD -----------------------');
  Log.Add('Rp   (R profile)          : ' + F(RProfile, 3));
  Log.Add('Rwp  (R weighted profile) : ' + F(RWeightedProfile, 3));
  Log.Add('Rexp (R expected)         : ' + F(RExpected, 3));
  Log.Add('GOF                       : ' + F(GOF, 3));
  Log.Add('');
  Log.Add('----------------------- PARAMETRES GLOBAUX -----------------------');
  Log.Add('Decalage du zero (deg)     : ' + F(ZeroShift, 5));
  Log.Add('Deplacement echantillon    : ' + F(SpecimenDisplacement, 5));
  Log.Add('');
  Log.Add('----------------------------- PHASES -----------------------------');
  Log.Add('Nombre de phases : ' + IntToStr(PhaseCount));
  total := 0.0;
  for p := 0 to PhaseCount - 1 do
  begin
    Log.Add('');
    Log.Add('[' + IntToStr(p + 1) + '] ' + PhaseTitle(p));
    if not PhaseUsed(p) then
      Log.Add('    (phase desactivee)');
    Log.Add('    Groupe d''espace   : ' + PhaseSpaceGroup(p));
    Log.Add('    % massique        : ' + F(PhaseWeightPct(p), 2) + ' +/- ' + F(PhaseWeightESD(p), 2));
    Log.Add('    Facteur d''echelle : ' + Format('%e', [PhaseScale(p)]));
    Log.Add('    R-Bragg           : ' + F(PhaseRBragg(p), 3));
    for k := 0 to 5 do
      Log.Add('    ' + CellNames[k] + ': ' + F(PhaseCell(p, k), 5));
    Log.Add('    Volume (A3)       : ' + F(PhaseVolume(p), 3));
    for k := 0 to 2 do
      Log.Add('    ' + UVWNames[k] + '                 : ' + F(PhaseProfileUVW(p, k), 5));
    if PhaseUsed(p) then
      total := total + PhaseWeightPct(p);
  end;
  Log.Add('');
  Log.Add('Somme des % massiques (phases actives) : ' + F(total, 2));
  Log.Add('');
  Log.Add('------------------------- AVERTISSEMENTS -------------------------');
  if GOF > 2.0 then
    Log.Add('! GOF > 2 : ajustement a verifier.');
  for p := 0 to PhaseCount - 1 do
    if PhaseUsed(p) and (PhaseWeightPct(p) < 1.0) then
      Log.Add('! ' + PhaseTitle(p) + ' < 1 % massique : presence a confirmer.');
  Log.Add('==================================================================');
end;

//======================================================================//
// Programme principal                                                  //
//======================================================================//

begin
  CheckIfDocExists;
  BaseName := GetUserFolder + CleanName(ScanShortName);
  Csv := TStringList.Create;
  Log := TStringList.Create;
  try
    if EXPORT_POINTS then
    begin
      WriteProfile;
      Csv.SaveToFile(BaseName + '_profil.csv');
    end;
    WriteLog;
    Log.SaveToFile(BaseName + '_rietveld.log');
    ShowMessage('Export termine :' + #13#10 + BaseName + '_rietveld.log' +
                #13#10 + BaseName + '_profil.csv');
  finally
    Csv.Free;
    Log.Free;
  end;
end.
