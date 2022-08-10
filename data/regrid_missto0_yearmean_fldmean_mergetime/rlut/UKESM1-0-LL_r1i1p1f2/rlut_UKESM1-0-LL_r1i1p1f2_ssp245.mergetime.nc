CDF   �   
      time       bnds      lon       lat          0   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       �UKESM1.0-LL (2018): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: UKCA-StratTrop
land: JULES-ES-1.0
landIce: none
ocean: NEMO-HadGEM3-GO6.0 (eORCA1 tripolar primarily 1 deg with meridional refinement down to 1/3 degree in the tropics; 360 x 330 longitude/latitude; 75 levels; top grid cell 0-1 m)
ocnBgchem: MEDUSA2
seaIce: CICE-HadGEM3-GSI8 (eORCA1 tripolar primarily 1 deg; 360 x 330 longitude/latitude)   institution       BMet Office Hadley Centre, Fitzroy Road, Exeter, Devon, EX1 3PB, UK     activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         A�        creation_date         2019-04-05T16:01:38Z   
cv_version        6.2.20.1   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Nhttps://furtherinfo.es-doc.org/CMIP6.MOHC.UKESM1-0-LL.historical.none.r1i1p1f2     grid      -Native N96 grid; 192 x 144 longitude/latitude      
grid_label        gn     history      =Wed Aug 10 15:20:42 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rlut/UKESM1-0-LL_r1i1p1f2/rlut_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp245.r1i1p1f2.Amon.rlut.gn.v20190507/rlut_Amon_UKESM1-0-LL_ssp245_r1i1p1f2_gn_201501-204912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/UKESM1-0-LL_r1i1p1f2/CMIP6.ScenarioMIP.MOHC.UKESM1-0-LL.ssp245.r1i1p1f2.Amon.rlut.gn.v20190507/rlut_Amon_UKESM1-0-LL_ssp245_r1i1p1f2_gn_205001-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rlut/UKESM1-0-LL_r1i1p1f2/rlut_UKESM1-0-LL_r1i1p1f2_ssp245.mergetime.nc
Wed Aug 10 15:20:41 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rlut.gn.v20190406/rlut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rlut.gn.v20190406/rlut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_195001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rlut/UKESM1-0-LL_r1i1p1f2/rlut_UKESM1-0-LL_r1i1p1f2_historical.mergetime.nc
Fri Apr 08 07:06:41 2022: cdo -O -s -selname,rlut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rlut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rlut.gn.v20190406/rlut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rlut.gn.v20190406/rlut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 07:06:37 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rlut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rlut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rlut.gn.v20190406/rlut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rlut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rlut.gn.v20190406/rlut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rlut/UKESM1-0-LL_r1i1p1f2/CMIP6.CMIP.MOHC.UKESM1-0-LL.historical.r1i1p1f2.Amon.rlut.gn.v20190406/rlut_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_185001-194912.bic_missto0.yearmean.nc
2019-04-05T15:50:03Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
2019-04-05T15:49:43Z MIP Convert v1.0.2, Python v2.7.12, Iris v1.13.0, Numpy v1.13.3, netcdftime v1.4.1.      initialization_index            institution_id        MOHC   mip_era       CMIP6      mo_runid      u-bc179    nominal_resolution        250 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      UKESM1-0-LL    parent_time_units         days since 1850-01-01-00-00-00     parent_variant_label      r1i1p1f2   physics_index               product       model-output   realization_index               realm         atmos      	source_id         UKESM1-0-LL    source_type       AOGCM AER BGC CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(13 December 2018) MD5:2b12b5db6db112aa8b8b0d6c1645b121      title         %UKESM1-0-LL output prepared for CMIP6      variable_id       rlut   variant_label         r1i1p1f2   license      XCMIP6 model data produced by the Met Office Hadley Centre is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file) and at https://ukesm.ac.uk/cmip6. The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   cmor_version      3.4.0      tracking_id       1hdl:21.14100/cb596009-b6f5-42df-8b78-3136f7182833      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      360_day    axis      T               �   	time_bnds                                 �   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               �   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               �   rlut                   
   standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       Iat the top of the atmosphere (to be compared with satellite measurements)      original_name         $mo: (stash: m01s03i332, lbproc: 128)   cell_measures         area: areacella    history       u2019-04-05T16:01:38Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               �                Aq���   Aq��P   Aq�P   Cp��Aq�6�   Aq�P   Aq��P   Cp�_Aq���   Aq��P   Aq��P   Cp�oAq��   Aq��P   Aq�dP   Cp�`Aq���   Aq�dP   Aq��P   Cq<�Aq���   Aq��P   Aq�FP   Cpb�Aq�k�   Aq�FP   Aq��P   Cq�Aq���   Aq��P   Aq�(P   Cq^\Aq�M�   Aq�(P   Aq��P   Cq�Aq���   Aq��P   Aq�
P   Cq%DAq�/�   Aq�
P   Aq�{P   Cp�)Aq���   Aq�{P   Aq��P   Cp��Aq��   Aq��P   Aq�]P   Cp�AqĂ�   Aq�]P   Aq��P   Cp�xAq���   Aq��P   Aq�?P   CqG�Aq�d�   Aq�?P   Aq˰P   Cpw�Aq���   Aq˰P   Aq�!P   Cp�VAq�F�   Aq�!P   AqВP   CpܒAqз�   AqВP   Aq�P   Cp��Aq�(�   Aq�P   Aq�tP   Cp�aAqՙ�   Aq�tP   Aq��P   Co��Aq�
�   Aq��P   Aq�VP   Cp�Aq�{�   Aq�VP   Aq��P   Cp��Aq���   Aq��P   Aq�8P   Cq��Aq�]�   Aq�8P   Aq�P   Cp��Aq���   Aq�P   Aq�P   Cp�ZAq�?�   Aq�P   Aq�P   Cp�IAq��   Aq�P   Aq��P   Cp�TAq�!�   Aq��P   Aq�mP   Cqo�Aq��   Aq�mP   Aq��P   Cp��Aq��   Aq��P   Aq�OP   Cp��Aq�t�   Aq�OP   Aq��P   Cp��Aq���   Aq��P   Aq�1P   Cp�SAq�V�   Aq�1P   Aq��P   CpX<Aq���   Aq��P   Aq�P   Co(Aq�8�   Aq�P   Aq��P   Co��Aq���   Aq��P   Aq��P   Cp�Aq��   Aq��P   ArfP   Cp4�Ar��   ArfP   Ar�P   Cp�dAr��   Ar�P   ArHP   Cp�Arm�   ArHP   Ar�P   Cpk�Ar��   Ar�P   Ar*P   Cq ^ArO�   Ar*P   Ar�P   Cq[#Ar��   Ar�P   ArP   Cp�Ar1�   ArP   Ar}P   CppAr��   Ar}P   Ar�P   Cp��Ar�   Ar�P   Ar_P   Cp��Ar��   Ar_P   Ar�P   Cp�(Ar��   Ar�P   ArAP   Cp��Arf�   ArAP   Ar�P   Cpw�Ar��   Ar�P   Ar!#P   Cp�EAr!H�   Ar!#P   Ar#�P   Cp}�Ar#��   Ar#�P   Ar&P   Cp<Ar&*�   Ar&P   Ar(vP   Cp�Ar(��   Ar(vP   Ar*�P   Cp&[Ar+�   Ar*�P   Ar-XP   Cp)Ar-}�   Ar-XP   Ar/�P   Cp'Ar/��   Ar/�P   Ar2:P   Cp��Ar2_�   Ar2:P   Ar4�P   Cp� Ar4��   Ar4�P   Ar7P   Cp�/Ar7A�   Ar7P   Ar9�P   CpILAr9��   Ar9�P   Ar;�P   Cpp#Ar<#�   Ar;�P   Ar>oP   Cp jAr>��   Ar>oP   Ar@�P   CpPmArA�   Ar@�P   ArCQP   Cp�.ArCv�   ArCQP   ArE�P   Cp��ArE��   ArE�P   ArH3P   Cp�-ArHX�   ArH3P   ArJ�P   Cp]�ArJ��   ArJ�P   ArMP   Cp��ArM:�   ArMP   ArO�P   Cp�FArO��   ArO�P   ArQ�P   Cp��ArR�   ArQ�P   ArThP   Cp6�ArT��   ArThP   ArV�P   Cp5ArV��   ArV�P   ArYJP   Cp7ArYo�   ArYJP   Ar[�P   Cp��Ar[��   Ar[�P   Ar^,P   Cp�Ar^Q�   Ar^,P   Ar`�P   Cp��Ar`��   Ar`�P   ArcP   Co��Arc3�   ArcP   AreP   Cp$�Are��   AreP   Arg�P   CpCArh�   Arg�P   ArjaP   Cq%hArj��   ArjaP   Arl�P   Cp�Arl��   Arl�P   AroCP   Cp0Aroh�   AroCP   Arq�P   CpxArq��   Arq�P   Art%P   Cp>�ArtJ�   Art%P   Arv�P   Cp�Arv��   Arv�P   AryP   Cp�,Ary,�   AryP   Ar{xP   Cp�Ar{��   Ar{xP   Ar}�P   Co��Ar~�   Ar}�P   Ar�ZP   Cp�Ar��   Ar�ZP   Ar��P   Cp<�Ar���   Ar��P   Ar�<P   CpwAr�a�   Ar�<P   Ar��P   Cpa�Ar���   Ar��P   Ar�P   Cp2Ar�C�   Ar�P   Ar��P   Cp�Ar���   Ar��P   Ar� P   Cp NAr�%�   Ar� P   Ar�qP   Co��Ar���   Ar�qP   Ar��P   Cpk�Ar��   Ar��P   Ar�SP   Cp��Ar�x�   Ar�SP   Ar��P   CpSAr���   Ar��P   Ar�5P   Co��Ar�Z�   Ar�5P   Ar��P   Co��Ar���   Ar��P   Ar�P   Co��Ar�<�   Ar�P   Ar��P   Co�Ar���   Ar��P   Ar��P   Cp�Ar��   Ar��P   Ar�jP   Cpl�Ar���   Ar�jP   Ar��P   CpK�Ar� �   Ar��P   Ar�LP   Cp3DAr�q�   Ar�LP   Ar��P   Cp��Ar���   Ar��P   Ar�.P   Co� Ar�S�   Ar�.P   Ar��P   Cp)Ar���   Ar��P   Ar�P   Co�mAr�5�   Ar�P   Ar��P   Cp�Ar���   Ar��P   Ar��P   Co"Ar��   Ar��P   Ar�cP   CnO�Ar���   Ar�cP   Ar��P   CoAr���   Ar��P   Ar�EP   CoЌAr�j�   Ar�EP   ArĶP   Cn�wAr���   ArĶP   Ar�'P   Cn�_Ar�L�   Ar�'P   ArɘP   CorArɽ�   ArɘP   Ar�	P   Co�Ar�.�   Ar�	P   Ar�zP   Cn��ArΟ�   Ar�zP   Ar��P   Cn�Ar��   Ar��P   Ar�\P   Cn��ArӁ�   Ar�\P   Ar��P   Cnp�Ar���   Ar��P   Ar�>P   Cn�sAr�c�   Ar�>P   ArگP   Cn��Ar���   ArگP   Ar� P   Cn��Ar�E�   Ar� P   ArߑP   Coj�Ar߶�   ArߑP   Ar�P   Co"Ar�'�   Ar�P   Ar�sP   Cn��Ar��   Ar�sP   Ar��P   CouAr�	�   Ar��P   Ar�UP   Cn�bAr�z�   Ar�UP   Ar��P   Cn�:Ar���   Ar��P   Ar�7P   Cn�Ar�\�   Ar�7P   Ar�P   Cn��Ar���   Ar�P   Ar�P   Cn�kAr�>�   Ar�P   Ar��P   Co�Ar���   Ar��P   Ar��P   Cn�GAr� �   Ar��P   Ar�lP   Co1Ar���   Ar�lP   Ar��P   Co�^Ar��   Ar��P   Ar�NP   CnwkAr�s�   Ar�NP   As�P   Cm�As��   As�P   As0P   Cm�kAsU�   As0P   As�P   CoWAs��   As�P   As	P   Cn��As	7�   As	P   As�P   Co{As��   As�P   As�P   Co_&As�   As�P   AseP   Cp	�As��   AseP   As�P   Co�UAs��   As�P   AsGP   Co�%Asl�   AsGP   As�P   Co��As��   As�P   As)P   Cp��AsN�   As)P   As�P   Co�+As��   As�P   AsP   Co�lAs0�   AsP   As!|P   Co�cAs!��   As!|P   As#�P   Co�/As$�   As#�P   As&^P   Cp!As&��   As&^P   As(�P   Cp*�As(��   As(�P   As+@P   Cp6�As+e�   As+@P   As-�P   Cpm�As-��   As-�P   As0"P   CpuAs0G�   As0"P   As2�P   Cp{�As2��   As2�P   As5P   CpW�As5)�   As5P   As7uP   Cp�yAs7��   As7uP   As9�P   Cp��As:�   As9�P   As<WP   Cp��As<|�   As<WP   As>�P   Cp�=As>��   As>�P   AsA9P   Cp�!AsA^�   AsA9P   AsC�P   Cp�<AsC��   AsC�P   AsFP   CqSAsF@�   AsFP   AsH�P   Cp��AsH��   AsH�P   AsJ�P   CqAsK"�   AsJ�P   AsMnP   Cqy�AsM��   AsMnP   AsO�P   Cq�qAsP�   AsO�P   AsRPP   Cq�xAsRu�   AsRPP   AsT�P   Cq��AsT��   AsT�P   AsW2P   Cqb�AsWW�   AsW2P   AsY�P   Cq�"AsY��   AsY�P   As\P   Cq�1As\9�   As\P   As^�P   Cq�As^��   As^�P   As`�P   CrQ Asa�   As`�P   AscgP   CrPAsc��   AscgP   Ase�P   Cq�IAse��   Ase�P   AshIP   Cq�mAshn�   AshIP   Asj�P   Cq��Asj��   Asj�P   Asm+P   CrA�AsmP�   Asm+P   Aso�P   Cr�Aso��   Aso�P   AsrP   Cr�Asr2�   AsrP   Ast~P   Crg�Ast��   Ast~P   Asv�P   Cr�ZAsw�   Asv�P   Asy`P   Cr�IAsy��   Asy`P   As{�P   Cr�$As{��   As{�P   As~BP   Cr_�As~g�   As~BP   As��P   Cr�As���   As��P   As�$P   Cr�[As�I�   As�$P   As��P   Cr��As���   As��P   As�P   CsM�As�+�   As�P   As�wP   Cse�As���   As�wP   As��P   Cs�As��   As��P   As�YP   Cs9�As�~�   As�YP   As��P   Cs�As���   As��P   As�;P   CsAs�`�   As�;P   As��P   CsPhAs���   As��P   As�P   Cs��As�B�   As�P   As��P   Cs�iAs���   As��P   As��P   CstcAs�$�   As��P   As�pP   Cs� As���   As�pP   As��P   CtgAs��   As��P   As�RP   CtPAs�w�   As�RP   As��P   CsֳAs���   As��P   As�4P   Cs��As�Y�   As�4P   As��P   Ct��As���   As��P   As�P   Ct�As�;�   As�P   As��P   Cs��As���   As��P   As��P   Cs�)As��   As��P   As�iP   Cs��As���   As�iP   As��P   Ct2BAs���   As��P   As�KP   Ct;VAs�p�   As�KP   As��P   Ct=�As���   As��P   As�-P   Ct�As�R�   As�-P   AsP   Cu�As���   AsP   As�P   CtR>As�4�   As�P   AsǀP   Ct��Asǥ�   AsǀP   As��P   Ct�EAs��   As��P   As�bP   Ct�GAṡ�   As�bP   As��P   Ct�As���   As��P   As�DP   CuvlAs�i�   As�DP   AsӵP   Ct��As���   AsӵP   As�&P   Ct��As�K�   As�&P   AsؗP   Ct��Asؼ�   AsؗP   As�P   Cu�As�-�   As�P   As�yP   Cu��Asݞ�   As�yP   As��P   Cu�As��   As��P   As�[P   Ct�As��   As�[P   As��P   CuAs���   As��P   As�=P   Cu�As�b�   As�=P   As�P   Cu�}As���   As�P   As�P   Cu�1As�D�   As�P   As�P   Cu�[As��   As�P   As�P   CuTAs�&�   As�P   As�rP   Cu�PAs��   As�rP   As��P   Cu��As��   As��P   As�TP   Cu�As�y�   As�TP   As��P   Cu��As���   As��P   As�6P   Cv&�As�[�   As�6P   As��P   Cu�As���   As��P   AtP   Cu��At=�   AtP   At�P   CvkMAt��   At�P   At�P   Cu��At�   At�P   At	kP   Cu��