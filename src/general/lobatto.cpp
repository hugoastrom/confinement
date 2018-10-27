/*
 *                This source code is part of
 *
 *                          HelFEM
 *                             -
 * Finite element methods for electronic structure calculations on small systems
 *
 * Written by Susi Lehtola, 2018-
 * Copyright (c) 2018- Susi Lehtola
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 */
#include <armadillo>
#include <cfloat>

void lobatto_set(int order, arma::vec & xtab, arma::vec & weight)

/******************************************************************************/
/*
  Purpose:

    LOBATTO_SET sets abscissas and weights for Lobatto quadrature.

  Discussion:

    The integral:

      Integral ( -1 <= X <= 1 ) F(X) dX

    The quadrature rule:

      Sum ( 1 <= I <= ORDER ) WEIGHt[I) * F ( XTAb[I) )

    The quadrature rule will integrate exactly all polynomials up to
    X**(2*ORDER-3).

    The Lobatto rule is distinguished by the fact that both endpoints
    (-1 and 1) are always abscissas.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 April 2006

  Author:

    John Burkardt

  Reference:

    Milton Abramowitz, Irene Stegun,
    Handbook of Mathematical Functions,
    National Bureau of Standards, 1964,
    ISBN: 0-486-61272-4,
    LC: QA47.A34.

    Arthur Stroud, Don Secrest,
    Gaussian Quadrature Formulas,
    Prentice Hall, 1966,
    LC: QA299.4G3S7.

    Daniel Zwillinger, editor,
    CRC Standard Mathematical Tables and Formulae,
    30th Edition,
    CRC Press, 1996,
    ISBN: 0-8493-2479-3.

  Parameters:

    Input, int ORDER, the order.
    ORDER must be between 2 and 20.

    Output, double XTAB[ORDER], the abscissas.

    Output, double WEIGHT[ORDER], the weights.
*/
{
  xtab.resize(order);
  weight.resize(order);

  if ( order == 2 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =    1.0E+00;

    weight[0] =  1.0E+00;
    weight[1] =  1.0E+00;
  }
  else if ( order == 3 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =    0.0E+00;
    xtab[2] =    1.0E+00;

    weight[0] =  1.0 / 3.0E+00;
    weight[1] =  4.0 / 3.0E+00;
    weight[2] =  1.0 / 3.0E+00;
  }
  else if ( order == 4 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.447213595499957939281834733746E+00;
    xtab[2] =    0.447213595499957939281834733746E+00;
    xtab[3] =    1.0E+00;

    weight[0] =  1.0E+00 / 6.0E+00;
    weight[1] =  5.0E+00 / 6.0E+00;
    weight[2] =  5.0E+00 / 6.0E+00;
    weight[3] =  1.0E+00 / 6.0E+00;
  }
  else if ( order == 5 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.654653670707977143798292456247E+00;
    xtab[2] =    0.0E+00;
    xtab[3] =    0.654653670707977143798292456247E+00;
    xtab[4] =    1.0E+00;

    weight[0] =  9.0E+00 / 90.0E+00;
    weight[1] = 49.0E+00 / 90.0E+00;
    weight[2] = 64.0E+00 / 90.0E+00;
    weight[3] = 49.0E+00 / 90.0E+00;
    weight[4] =  9.0E+00 / 90.0E+00;
  }
  else if ( order == 6 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.765055323929464692851002973959E+00;
    xtab[2] =  - 0.285231516480645096314150994041E+00;
    xtab[3] =    0.285231516480645096314150994041E+00;
    xtab[4] =    0.765055323929464692851002973959E+00;
    xtab[5] =    1.0E+00;

    weight[0] =  0.066666666666666666666666666667E+00;
    weight[1] =  0.378474956297846980316612808212E+00;
    weight[2] =  0.554858377035486353016720525121E+00;
    weight[3] =  0.554858377035486353016720525121E+00;
    weight[4] =  0.378474956297846980316612808212E+00;
    weight[5] =  0.066666666666666666666666666667E+00;
  }
  else if ( order == 7 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.830223896278566929872032213967E+00;
    xtab[2] =  - 0.468848793470714213803771881909E+00;
    xtab[3] =    0.0E+00;
    xtab[4] =    0.468848793470714213803771881909E+00;
    xtab[5] =    0.830223896278566929872032213967E+00;
    xtab[6] =    1.0E+00;

    weight[0] =  0.476190476190476190476190476190E-01;
    weight[1] =  0.276826047361565948010700406290E+00;
    weight[2] =  0.431745381209862623417871022281E+00;
    weight[3] =  0.487619047619047619047619047619E+00;
    weight[4] =  0.431745381209862623417871022281E+00;
    weight[5] =  0.276826047361565948010700406290E+00;
    weight[6] =  0.476190476190476190476190476190E-01;
  }
  else if ( order == 8 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.871740148509606615337445761221E+00;
    xtab[2] =  - 0.591700181433142302144510731398E+00;
    xtab[3] =  - 0.209299217902478868768657260345E+00;
    xtab[4] =    0.209299217902478868768657260345E+00;
    xtab[5] =    0.591700181433142302144510731398E+00;
    xtab[6] =    0.871740148509606615337445761221E+00;
    xtab[7] =    1.0E+00;

    weight[0] =  0.357142857142857142857142857143E-01;
    weight[1] =  0.210704227143506039382991065776E+00;
    weight[2] =  0.341122692483504364764240677108E+00;
    weight[3] =  0.412458794658703881567052971402E+00;
    weight[4] =  0.412458794658703881567052971402E+00;
    weight[5] =  0.341122692483504364764240677108E+00;
    weight[6] =  0.210704227143506039382991065776E+00;
    weight[7] =  0.357142857142857142857142857143E-01;
  }
  else if ( order == 9 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.899757995411460157312345244418E+00;
    xtab[2] =  - 0.677186279510737753445885427091E+00;
    xtab[3] =  - 0.363117463826178158710752068709E+00;
    xtab[4] =    0.0E+00;
    xtab[5] =    0.363117463826178158710752068709E+00;
    xtab[6] =    0.677186279510737753445885427091E+00;
    xtab[7] =    0.899757995411460157312345244418E+00;
    xtab[8] =    1.0E+00;

    weight[0] =  0.277777777777777777777777777778E-01;
    weight[1] =  0.165495361560805525046339720029E+00;
    weight[2] =  0.274538712500161735280705618579E+00;
    weight[3] =  0.346428510973046345115131532140E+00;
    weight[4] =  0.371519274376417233560090702948E+00;
    weight[5] =  0.346428510973046345115131532140E+00;
    weight[6] =  0.274538712500161735280705618579E+00;
    weight[7] =  0.165495361560805525046339720029E+00;
    weight[8] =  0.277777777777777777777777777778E-01;
  }
  else if ( order == 10 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.919533908166458813828932660822E+00;
    xtab[2] =  - 0.738773865105505075003106174860E+00;
    xtab[3] =  - 0.477924949810444495661175092731E+00;
    xtab[4] =  - 0.165278957666387024626219765958E+00;
    xtab[5] =    0.165278957666387024626219765958E+00;
    xtab[6] =    0.477924949810444495661175092731E+00;
    xtab[7] =    0.738773865105505075003106174860E+00;
    xtab[8] =    0.919533908166458813828932660822E+00;
    xtab[9] =   1.0E+00;

    weight[0] =  0.222222222222222222222222222222E-01;
    weight[1] =  0.133305990851070111126227170755E+00;
    weight[2] =  0.224889342063126452119457821731E+00;
    weight[3] =  0.292042683679683757875582257374E+00;
    weight[4] =  0.327539761183897456656510527917E+00;
    weight[5] =  0.327539761183897456656510527917E+00;
    weight[6] =  0.292042683679683757875582257374E+00;
    weight[7] =  0.224889342063126452119457821731E+00;
    weight[8] =  0.133305990851070111126227170755E+00;
    weight[9] = 0.222222222222222222222222222222E-01;
  }
  else if ( order == 11 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.934001430408059134332274136099E+00;
    xtab[2] =  - 0.784483473663144418622417816108E+00;
    xtab[3] =  - 0.565235326996205006470963969478E+00;
    xtab[4] =  - 0.295758135586939391431911515559E+00;
    xtab[5] =    0.0E+00;
    xtab[6] =    0.295758135586939391431911515559E+00;
    xtab[7] =    0.565235326996205006470963969478E+00;
    xtab[8] =    0.784483473663144418622417816108E+00;
    xtab[9] =   0.934001430408059134332274136099E+00;
    xtab[10] =   1.0E+00;

    weight[0] =  0.181818181818181818181818181818E-01;
    weight[1] =  0.109612273266994864461403449580E+00;
    weight[2] =  0.187169881780305204108141521899E+00;
    weight[3] =  0.248048104264028314040084866422E+00;
    weight[4] =  0.286879124779008088679222403332E+00;
    weight[5] =  0.300217595455690693785931881170E+00;
    weight[6] =  0.286879124779008088679222403332E+00;
    weight[7] =  0.248048104264028314040084866422E+00;
    weight[8] =  0.187169881780305204108141521899E+00;
    weight[9] = 0.109612273266994864461403449580E+00;
    weight[10] = 0.181818181818181818181818181818E-01;
  }
  else if ( order == 12 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.944899272222882223407580138303E+00;
    xtab[2] =  - 0.819279321644006678348641581717E+00;
    xtab[3] =  - 0.632876153031869677662404854444E+00;
    xtab[4] =  - 0.399530940965348932264349791567E+00;
    xtab[5] =  - 0.136552932854927554864061855740E+00;
    xtab[6] =    0.136552932854927554864061855740E+00;
    xtab[7] =    0.399530940965348932264349791567E+00;
    xtab[8] =    0.632876153031869677662404854444E+00;
    xtab[9] =   0.819279321644006678348641581717E+00;
    xtab[10] =   0.944899272222882223407580138303E+00;
    xtab[11] =   1.0E+00;

    weight[0] =  0.151515151515151515151515151515E-01;
    weight[1] =  0.916845174131961306683425941341E-01;
    weight[2] =  0.157974705564370115164671062700E+00;
    weight[3] =  0.212508417761021145358302077367E+00;
    weight[4] =  0.251275603199201280293244412148E+00;
    weight[5] =  0.271405240910696177000288338500E+00;
    weight[6] =  0.271405240910696177000288338500E+00;
    weight[7] =  0.251275603199201280293244412148E+00;
    weight[8] =  0.212508417761021145358302077367E+00;
    weight[9] = 0.157974705564370115164671062700E+00;
    weight[10] = 0.916845174131961306683425941341E-01;
    weight[11] = 0.151515151515151515151515151515E-01;
  }
  else if ( order == 13 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.953309846642163911896905464755E+00;
    xtab[2] =  - 0.846347564651872316865925607099E+00;
    xtab[3] =  - 0.686188469081757426072759039566E+00;
    xtab[4] =  - 0.482909821091336201746937233637E+00;
    xtab[5] =  - 0.249286930106239992568673700374E+00;
    xtab[6] =    0.0E+00;
    xtab[7] =    0.249286930106239992568673700374E+00;
    xtab[8] =    0.482909821091336201746937233637E+00;
    xtab[9] =   0.686188469081757426072759039566E+00;
    xtab[10] =   0.846347564651872316865925607099E+00;
    xtab[11] =   0.953309846642163911896905464755E+00;
    xtab[12] =   1.0E+00;

    weight[0] =  0.128205128205128205128205128205E-01;
    weight[1] =  0.778016867468189277935889883331E-01;
    weight[2] =  0.134981926689608349119914762589E+00;
    weight[3] =  0.183646865203550092007494258747E+00;
    weight[4] =  0.220767793566110086085534008379E+00;
    weight[5] =  0.244015790306676356458578148360E+00;
    weight[6] =  0.251930849333446736044138641541E+00;
    weight[7] =  0.244015790306676356458578148360E+00;
    weight[8] =  0.220767793566110086085534008379E+00;
    weight[9] = 0.183646865203550092007494258747E+00;
    weight[10] = 0.134981926689608349119914762589E+00;
    weight[11] = 0.778016867468189277935889883331E-01;
    weight[12] = 0.128205128205128205128205128205E-01;
  }
  else if ( order == 14 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.959935045267260901355100162015E+00;
    xtab[2] =  - 0.867801053830347251000220202908E+00;
    xtab[3] =  - 0.728868599091326140584672400521E+00;
    xtab[4] =  - 0.550639402928647055316622705859E+00;
    xtab[5] =  - 0.342724013342712845043903403642E+00;
    xtab[6] =  - 0.116331868883703867658776709736E+00;
    xtab[7] =    0.116331868883703867658776709736E+00;
    xtab[8] =    0.342724013342712845043903403642E+00;
    xtab[9] =   0.550639402928647055316622705859E+00;
    xtab[10] =   0.728868599091326140584672400521E+00;
    xtab[11] =   0.867801053830347251000220202908E+00;
    xtab[12] =   0.959935045267260901355100162015E+00;
    xtab[13] =   1.0E+00;

    weight[0] =  0.109890109890109890109890109890E-01;
    weight[1] =  0.668372844976812846340706607461E-01;
    weight[2] =  0.116586655898711651540996670655E+00;
    weight[3] =  0.160021851762952142412820997988E+00;
    weight[4] =  0.194826149373416118640331778376E+00;
    weight[5] =  0.219126253009770754871162523954E+00;
    weight[6] =  0.231612794468457058889628357293E+00;
    weight[7] =  0.231612794468457058889628357293E+00;
    weight[8] =  0.219126253009770754871162523954E+00;
    weight[9] = 0.194826149373416118640331778376E+00;
    weight[10] = 0.160021851762952142412820997988E+00;
    weight[11] = 0.116586655898711651540996670655E+00;
    weight[12] = 0.668372844976812846340706607461E-01;
    weight[13] = 0.109890109890109890109890109890E-01;
  }
  else if ( order == 15 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.965245926503838572795851392070E+00;
    xtab[2] =  - 0.885082044222976298825401631482E+00;
    xtab[3] =  - 0.763519689951815200704118475976E+00;
    xtab[4] =  - 0.606253205469845711123529938637E+00;
    xtab[5] =  - 0.420638054713672480921896938739E+00;
    xtab[6] =  - 0.215353955363794238225679446273E+00;
    xtab[7] =    0.0E+00;
    xtab[8] =    0.215353955363794238225679446273E+00;
    xtab[9] =   0.420638054713672480921896938739E+00;
    xtab[10] =   0.606253205469845711123529938637E+00;
    xtab[11] =   0.763519689951815200704118475976E+00;
    xtab[12] =   0.885082044222976298825401631482E+00;
    xtab[13] =   0.965245926503838572795851392070E+00;
    xtab[14] =   1.0E+00;

    weight[0] =  0.952380952380952380952380952381E-02;
    weight[1] =  0.580298930286012490968805840253E-01;
    weight[2] =  0.101660070325718067603666170789E+00;
    weight[3] =  0.140511699802428109460446805644E+00;
    weight[4] =  0.172789647253600949052077099408E+00;
    weight[5] =  0.196987235964613356092500346507E+00;
    weight[6] =  0.211973585926820920127430076977E+00;
    weight[7] =  0.217048116348815649514950214251E+00;
    weight[8] =  0.211973585926820920127430076977E+00;
    weight[9] = 0.196987235964613356092500346507E+00;
    weight[10] = 0.172789647253600949052077099408E+00;
    weight[11] = 0.140511699802428109460446805644E+00;
    weight[12] = 0.101660070325718067603666170789E+00;
    weight[13] = 0.580298930286012490968805840253E-01;
    weight[14] = 0.952380952380952380952380952381E-02;
  }
  else if ( order == 16 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.969568046270217932952242738367E+00;
    xtab[2] =  - 0.899200533093472092994628261520E+00;
    xtab[3] =  - 0.792008291861815063931088270963E+00;
    xtab[4] =  - 0.652388702882493089467883219641E+00;
    xtab[5] =  - 0.486059421887137611781890785847E+00;
    xtab[6] =  - 0.299830468900763208098353454722E+00;
    xtab[7] =  - 0.101326273521949447843033005046E+00;
    xtab[8] =    0.101326273521949447843033005046E+00;
    xtab[9] =   0.299830468900763208098353454722E+00;
    xtab[10] =   0.486059421887137611781890785847E+00;
    xtab[11] =   0.652388702882493089467883219641E+00;
    xtab[12] =   0.792008291861815063931088270963E+00;
    xtab[13] =   0.899200533093472092994628261520E+00;
    xtab[14] =   0.969568046270217932952242738367E+00;
    xtab[15] =   1.0E+00;

    weight[0] =  0.833333333333333333333333333333E-02;
    weight[1] =  0.508503610059199054032449195655E-01;
    weight[2] =  0.893936973259308009910520801661E-01;
    weight[3] =  0.124255382132514098349536332657E+00;
    weight[4] =  0.154026980807164280815644940485E+00;
    weight[5] =  0.177491913391704125301075669528E+00;
    weight[6] =  0.193690023825203584316913598854E+00;
    weight[7] =  0.201958308178229871489199125411E+00;
    weight[8] =  0.201958308178229871489199125411E+00;
    weight[9] = 0.193690023825203584316913598854E+00;
    weight[10] = 0.177491913391704125301075669528E+00;
    weight[11] = 0.154026980807164280815644940485E+00;
    weight[12] = 0.124255382132514098349536332657E+00;
    weight[13] = 0.893936973259308009910520801661E-01;
    weight[14] = 0.508503610059199054032449195655E-01;
    weight[15] = 0.833333333333333333333333333333E-02;
  }
  else if ( order == 17 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.973132176631418314156979501874E+00;
    xtab[2] =  - 0.910879995915573595623802506398E+00;
    xtab[3] =  - 0.815696251221770307106750553238E+00;
    xtab[4] =  - 0.691028980627684705394919357372E+00;
    xtab[5] =  - 0.541385399330101539123733407504E+00;
    xtab[6] =  - 0.372174433565477041907234680735E+00;
    xtab[7] =  - 0.189511973518317388304263014753E+00;
    xtab[8] =    0.0E+00;
    xtab[9] =   0.189511973518317388304263014753E+00;
    xtab[10] =   0.372174433565477041907234680735E+00;
    xtab[11] =   0.541385399330101539123733407504E+00;
    xtab[12] =   0.691028980627684705394919357372E+00;
    xtab[13] =   0.815696251221770307106750553238E+00;
    xtab[14] =   0.910879995915573595623802506398E+00;
    xtab[15] =   0.973132176631418314156979501874E+00;
    xtab[16] =   1.0E+00;

    weight[0] =  0.735294117647058823529411764706E-02;
    weight[1] =  0.449219405432542096474009546232E-01;
    weight[2] =  0.791982705036871191902644299528E-01;
    weight[3] =  0.110592909007028161375772705220E+00;
    weight[4] =  0.137987746201926559056201574954E+00;
    weight[5] =  0.160394661997621539516328365865E+00;
    weight[6] =  0.177004253515657870436945745363E+00;
    weight[7] =  0.187216339677619235892088482861E+00;
    weight[8] =  0.190661874753469433299407247028E+00;
    weight[9] = 0.187216339677619235892088482861E+00;
    weight[10] = 0.177004253515657870436945745363E+00;
    weight[11] = 0.160394661997621539516328365865E+00;
    weight[12] = 0.137987746201926559056201574954E+00;
    weight[13] = 0.110592909007028161375772705220E+00;
    weight[14] = 0.791982705036871191902644299528E-01;
    weight[15] = 0.449219405432542096474009546232E-01;
    weight[16] = 0.735294117647058823529411764706E-02;
  }
  else if ( order == 18 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.976105557412198542864518924342E+00;
    xtab[2] =  - 0.920649185347533873837854625431E+00;
    xtab[3] =  - 0.835593535218090213713646362328E+00;
    xtab[4] =  - 0.723679329283242681306210365302E+00;
    xtab[5] =  - 0.588504834318661761173535893194E+00;
    xtab[6] =  - 0.434415036912123975342287136741E+00;
    xtab[7] =  - 0.266362652878280984167665332026E+00;
    xtab[8] =  - 0.897490934846521110226450100886E-01;
    xtab[9] =   0.897490934846521110226450100886E-01;
    xtab[10] =   0.266362652878280984167665332026E+00;
    xtab[11] =   0.434415036912123975342287136741E+00;
    xtab[12] =   0.588504834318661761173535893194E+00;
    xtab[13] =   0.723679329283242681306210365302E+00;
    xtab[14] =   0.835593535218090213713646362328E+00;
    xtab[15] =   0.920649185347533873837854625431E+00;
    xtab[16] =   0.976105557412198542864518924342E+00;
    xtab[17] =   1.0E+00;

    weight[0] =  0.653594771241830065359477124183E-02;
    weight[1] =  0.399706288109140661375991764101E-01;
    weight[2] =  0.706371668856336649992229601678E-01;
    weight[3] =  0.990162717175028023944236053187E-01;
    weight[4] =  0.124210533132967100263396358897E+00;
    weight[5] =  0.145411961573802267983003210494E+00;
    weight[6] =  0.161939517237602489264326706700E+00;
    weight[7] =  0.173262109489456226010614403827E+00;
    weight[8] =  0.179015863439703082293818806944E+00;
    weight[9] = 0.179015863439703082293818806944E+00;
    weight[10] = 0.173262109489456226010614403827E+00;
    weight[11] = 0.161939517237602489264326706700E+00;
    weight[12] = 0.145411961573802267983003210494E+00;
    weight[13] = 0.124210533132967100263396358897E+00;
    weight[14] = 0.990162717175028023944236053187E-01;
    weight[15] = 0.706371668856336649992229601678E-01;
    weight[16] = 0.399706288109140661375991764101E-01;
    weight[17] = 0.653594771241830065359477124183E-02;
  }
  else if ( order == 19 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.978611766222080095152634063110E+00;
    xtab[2] =  - 0.928901528152586243717940258797E+00;
    xtab[3] =  - 0.852460577796646093085955970041E+00;
    xtab[4] =  - 0.751494202552613014163637489634E+00;
    xtab[5] =  - 0.628908137265220497766832306229E+00;
    xtab[6] =  - 0.488229285680713502777909637625E+00;
    xtab[7] =  - 0.333504847824498610298500103845E+00;
    xtab[8] =  - 0.169186023409281571375154153445E+00;
    xtab[9] =   0.0E+00;
    xtab[10] =   0.169186023409281571375154153445E+00;
    xtab[11] =   0.333504847824498610298500103845E+00;
    xtab[12] =   0.488229285680713502777909637625E+00;
    xtab[13] =   0.628908137265220497766832306229E+00;
    xtab[14] =   0.751494202552613014163637489634E+00;
    xtab[15] =   0.852460577796646093085955970041E+00;
    xtab[16] =   0.928901528152586243717940258797E+00;
    xtab[17] =   0.978611766222080095152634063110E+00;
    xtab[18] =   1.0E+00;

    weight[0] =  0.584795321637426900584795321637E-02;
    weight[1] =  0.357933651861764771154255690351E-01;
    weight[2] =  0.633818917626297368516956904183E-01;
    weight[3] =  0.891317570992070844480087905562E-01;
    weight[4] =  0.112315341477305044070910015464E+00;
    weight[5] =  0.132267280448750776926046733910E+00;
    weight[6] =  0.148413942595938885009680643668E+00;
    weight[7] =  0.160290924044061241979910968184E+00;
    weight[8] =  0.167556584527142867270137277740E+00;
    weight[9] = 0.170001919284827234644672715617E+00;
    weight[10] = 0.167556584527142867270137277740E+00;
    weight[11] = 0.160290924044061241979910968184E+00;
    weight[12] = 0.148413942595938885009680643668E+00;
    weight[13] = 0.132267280448750776926046733910E+00;
    weight[14] = 0.112315341477305044070910015464E+00;
    weight[15] = 0.891317570992070844480087905562E-01;
    weight[16] = 0.633818917626297368516956904183E-01;
    weight[17] = 0.357933651861764771154255690351E-01;
    weight[18] = 0.584795321637426900584795321637E-02;
  }
  else if ( order == 20 )
  {
    xtab[0] =  - 1.0E+00;
    xtab[1] =  - 0.980743704893914171925446438584E+00;
    xtab[2] =  - 0.935934498812665435716181584931E+00;
    xtab[3] =  - 0.866877978089950141309847214616E+00;
    xtab[4] =  - 0.775368260952055870414317527595E+00;
    xtab[5] =  - 0.663776402290311289846403322971E+00;
    xtab[6] =  - 0.534992864031886261648135961829E+00;
    xtab[7] =  - 0.392353183713909299386474703816E+00;
    xtab[8] =  - 0.239551705922986495182401356927E+00;
    xtab[9] = - 0.805459372388218379759445181596E-01;
    xtab[10] =   0.805459372388218379759445181596E-01;
    xtab[11] =   0.239551705922986495182401356927E+00;
    xtab[12] =   0.392353183713909299386474703816E+00;
    xtab[13] =   0.534992864031886261648135961829E+00;
    xtab[14] =   0.663776402290311289846403322971E+00;
    xtab[15] =   0.775368260952055870414317527595E+00;
    xtab[16] =   0.866877978089950141309847214616E+00;
    xtab[17] =   0.935934498812665435716181584931E+00;
    xtab[18] =   0.980743704893914171925446438584E+00;
    xtab[19] =   1.0E+00;

    weight[0] =  0.526315789473684210526315789474E-02;
    weight[1] =  0.322371231884889414916050281173E-01;
    weight[2] =  0.571818021275668260047536271732E-01;
    weight[3] =  0.806317639961196031447768461137E-01;
    weight[4] =  0.101991499699450815683781205733E+00;
    weight[5] =  0.120709227628674725099429705002E+00;
    weight[6] =  0.136300482358724184489780792989E+00;
    weight[7] =  0.148361554070916825814713013734E+00;
    weight[8] =  0.156580102647475487158169896794E+00;
    weight[9] = 0.160743286387845749007726726449E+00;
    weight[10] = 0.160743286387845749007726726449E+00;
    weight[11] = 0.156580102647475487158169896794E+00;
    weight[12] = 0.148361554070916825814713013734E+00;
    weight[13] = 0.136300482358724184489780792989E+00;
    weight[14] = 0.120709227628674725099429705002E+00;
    weight[15] = 0.101991499699450815683781205733E+00;
    weight[16] = 0.806317639961196031447768461137E-01;
    weight[17] = 0.571818021275668260047536271732E-01;
    weight[18] = 0.322371231884889414916050281173E-01;
    weight[19] = 0.526315789473684210526315789474E-02;
  }
  else
  {
    throw std::domain_error("Legal values for lobatto_set are between 2 and 20.\n");
  }
}


void lobatto_compute (int n, arma::vec & x, arma::vec & w)

/******************************************************************************/
/*
  Purpose:

    LOBATTO_COMPUTE computes a Lobatto quadrature rule.

  Discussion:

    The integral:

      Integral ( -1 <= X <= 1 ) F(X) dX

    The quadrature rule:

      Sum ( 1 <= I <= N ) WEIGHT(I) * F ( XTAB(I) )

    The quadrature rule will integrate exactly all polynomials up to
    X**(2*N-3).

    The Lobatto rule is distinguished by the fact that both endpoints
    (-1 and 1) are always abscissas.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    04 February 2007

  Author:

    Original MATLAB version by Greg von Winckel.
    C version by John Burkardt.

  Reference:

    Milton Abramowitz, Irene Stegun,
    Handbook of Mathematical Functions,
    National Bureau of Standards, 1964,
    ISBN: 0-486-61272-4,
    LC: QA47.A34.

    Claudio Canuto, Yousuff Hussaini, Alfio Quarteroni, Thomas Zang,
    Spectral Methods in Fluid Dynamics,
    Springer, 1993,
    ISNB13: 978-3540522058,
    LC: QA377.S676.

    Arthur Stroud, Don Secrest,
    Gaussian Quadrature Formulas,
    Prentice Hall, 1966,
    LC: QA299.4G3S7.

    Daniel Zwillinger, editor,
    CRC Standard Mathematical Tables and Formulae,
    30th Edition,
    CRC Press, 1996,
    ISBN: 0-8493-2479-3.

  Parameters:

    Input, int N, the order.
    N must be at least 2.

    Output, double X[N], the abscissas.

    Output, double W[N], the weights.
*/
{
  int i;
  int j;
  double test, error;
  double tolerance;

  if ( n < 2 )
  {
    std::ostringstream oss;
    oss << "Lobatto called with n="<<n<<", but n>=2 is required.\n";
    throw std::runtime_error(oss.str());
  }

  if(n<20) {
    // Use tabled weights and nodes
    lobatto_set(n,x,w);
    return;
  }

  // Resize tables to correct length
  x.resize(n);
  w.resize(n);

  tolerance = 100.0 * DBL_EPSILON;
/*
  Initial estimate for the abscissas is the Chebyshev-Gauss-Lobatto nodes.
*/
  for ( i = 0; i < n; i++ )
  {
    x[i] = cos ( M_PI * ( double ) ( i ) / ( double ) ( n - 1 ) );
  }

   double xold[n];
   double p[n*n];

  do
  {
    for ( i = 0; i < n; i++ )
    {
      xold[i] = x[i];
    }
    for ( i = 0; i < n; i++ )
    {
      p[i+0*n] = 1.0;
    }
    for ( i = 0; i < n; i++ )
    {
      p[i+1*n] = x[i];
    }

    for ( j = 2; j <= n-1; j++ )
    {
      for ( i = 0; i < n; i++)
      {
        p[i+j*n] = ( ( double ) ( 2 * j - 1 ) * x[i] * p[i+(j-1)*n]
                   + ( double ) (   - j + 1 ) *        p[i+(j-2)*n] )
                   / ( double ) (     j     );
      }
    }

    for ( i = 0; i < n; i++ )
    {
      x[i] = xold[i] - ( x[i] * p[i+(n-1)*n] - p[i+(n-2)*n] )
           / ( ( double ) ( n ) * p[i+(n-1)*n] );
    }

    error = 0.0;
    for ( i = 0; i < n; i++ )
    {
      test = fabs(x[i] - xold[i]);
      if(test>error)
	error=test;
    }

  } while ( tolerance < error );

  // Reverse order of x
  x = arma::reverse(x);

  for ( i = 0; i < n; i++ )
  {
    w[i] = 2.0 / ( ( double ) ( ( n - 1 ) * n ) * pow ( p[i+(n-1)*n], 2 ) );
  }
}
/******************************************************************************/

