CDF  �   
      time       bnds      lon       lat          .   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       EMRI-ESM2.0 (2017): 
aerosol: MASINGAR mk2r4 (TL95; 192 x 96 longitude/latitude; 80 levels; top level 0.01 hPa)
atmos: MRI-AGCM3.5 (TL159; 320 x 160 longitude/latitude; 80 levels; top level 0.01 hPa)
atmosChem: MRI-CCM2.1 (T42; 128 x 64 longitude/latitude; 80 levels; top level 0.01 hPa)
land: HAL 1.0
landIce: none
ocean: MRI.COM4.4 (tripolar primarily 0.5 deg latitude/1 deg longitude with meridional refinement down to 0.3 deg within 10 degrees north and south of the equator; 360 x 364 longitude/latitude; 61 levels; top grid cell 0-2 m)
ocnBgchem: MRI.COM4.4
seaIce: MRI.COM4.4      institution       CMeteorological Research Institute, Tsukuba, Ibaraki 305-0052, Japan    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-02-20T02:45:37Z   data_specs_version        01.00.29   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Lhttps://furtherinfo.es-doc.org/CMIP6.MRI.MRI-ESM2-0.historical.none.r1i1p1f1   grid      7native atmosphere TL159 gaussian grid (160x320 latxlon)    
grid_label        gn     history      
�Wed Aug 10 15:20:32 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rlut/MRI-ESM2-0_r1i1p1f1/rlut_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp126.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_MRI-ESM2-0_ssp126_r1i1p1f1_gn_201501-210012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.ScenarioMIP.MRI.MRI-ESM2-0.ssp126.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_MRI-ESM2-0_ssp126_r1i1p1f1_gn_210101-230012.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rlut/MRI-ESM2-0_r1i1p1f1/rlut_MRI-ESM2-0_r1i1p1f1_ssp126.mergetime.nc
Wed Aug 10 15:20:32 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rlut/MRI-ESM2-0_r1i1p1f1/rlut_MRI-ESM2-0_r1i1p1f1_historical.mergetime.nc
Fri Apr 08 07:01:19 2022: cdo -O -s -selname,rlut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 07:01:15 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rlut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rlut/MRI-ESM2-0_r1i1p1f1/CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.rlut.gn.v20190222/rlut_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc
2019-02-20T02:45:37Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.;
Output from run-Dr060_historical_101 (sfc_avr_mon.ctl)      initialization_index            institution_id        MRI    mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
MRI-ESM2-0     parent_time_units         days since 1850-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      	source_id         
MRI-ESM2-0     source_type       AOGCM AER CHEM     sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        ECreation Date:(14 December 2018) MD5:b2d32d1a0d9b196411429c8895329d8f      title         $MRI-ESM2-0 output prepared for CMIP6   variable_id       rlut   variant_label         r1i1p1f1   license      CMIP6 model data produced by MRI is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.    cmor_version      3.4.0      tracking_id       1hdl:21.14100/424f453d-9b80-4d60-9b9d-ec646b7cf434      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               H   	time_bnds                                 P   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               8   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               @   rlut                   
   standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       Iat the top of the atmosphere (to be compared with satellite measurements)      original_name         ULWT   cell_measures         area: areacella    history       r2019-02-20T02:45:37Z altered by CMOR: replaced missing value flag (-9.99e+33) with standard missing value (1e+20).              `                Aq���   Aq��P   Aq�P   Cq�Aq�6�   Aq�P   Aq��P   Cr�Aq���   Aq��P   Aq��P   Cq��Aq��   Aq��P   Aq�dP   Cq[�Aq���   Aq�dP   Aq��P   CqyiAq���   Aq��P   Aq�FP   Cq�Aq�k�   Aq�FP   Aq��P   Cq��Aq���   Aq��P   Aq�(P   CqmAq�M�   Aq�(P   Aq��P   Cqu�Aq���   Aq��P   Aq�
P   Cq�OAq�/�   Aq�
P   Aq�{P   Cq��Aq���   Aq�{P   Aq��P   CqY�Aq��   Aq��P   Aq�]P   Cq#/AqĂ�   Aq�]P   Aq��P   Cq��Aq���   Aq��P   Aq�?P   CqaAq�d�   Aq�?P   Aq˰P   Cqd�Aq���   Aq˰P   Aq�!P   Cq[�Aq�F�   Aq�!P   AqВP   Cq\�Aqз�   AqВP   Aq�P   Cq�Aq�(�   Aq�P   Aq�tP   Cqa�Aqՙ�   Aq�tP   Aq��P   Cq;"Aq�
�   Aq��P   Aq�VP   Cq��Aq�{�   Aq�VP   Aq��P   Cq,�Aq���   Aq��P   Aq�8P   CpŗAq�]�   Aq�8P   Aq�P   Cp�FAq���   Aq�P   Aq�P   Cq֥Aq�?�   Aq�P   Aq�P   CqTEAq��   Aq�P   Aq��P   Cq7�Aq�!�   Aq��P   Aq�mP   Cq��Aq��   Aq�mP   Aq��P   CqX@Aq��   Aq��P   Aq�OP   Cq<�Aq�t�   Aq�OP   Aq��P   Cq��Aq���   Aq��P   Aq�1P   Cq�vAq�V�   Aq�1P   Aq��P   CpܱAq���   Aq��P   Aq�P   Cp,�Aq�8�   Aq�P   Aq��P   Cp{�Aq���   Aq��P   Aq��P   Cp��Aq��   Aq��P   ArfP   Cp�eAr��   ArfP   Ar�P   Cq,7Ar��   Ar�P   ArHP   Cq�Arm�   ArHP   Ar�P   Cp�6Ar��   Ar�P   Ar*P   CqX�ArO�   Ar*P   Ar�P   Cq)|Ar��   Ar�P   ArP   Cqj�Ar1�   ArP   Ar}P   CqwDAr��   Ar}P   Ar�P   Cq��Ar�   Ar�P   Ar_P   Cq�<Ar��   Ar_P   Ar�P   Cqi�Ar��   Ar�P   ArAP   Cq�&Arf�   ArAP   Ar�P   Cqy�Ar��   Ar�P   Ar!#P   Cp��Ar!H�   Ar!#P   Ar#�P   Cq.rAr#��   Ar#�P   Ar&P   CqO�Ar&*�   Ar&P   Ar(vP   Cp��Ar(��   Ar(vP   Ar*�P   CpڇAr+�   Ar*�P   Ar-XP   Cqt!Ar-}�   Ar-XP   Ar/�P   Cq�Ar/��   Ar/�P   Ar2:P   Cp��Ar2_�   Ar2:P   Ar4�P   CqFAr4��   Ar4�P   Ar7P   CqPAr7A�   Ar7P   Ar9�P   Cq~Ar9��   Ar9�P   Ar;�P   Cp�7Ar<#�   Ar;�P   Ar>oP   Cp��Ar>��   Ar>oP   Ar@�P   Cp��ArA�   Ar@�P   ArCQP   Cq�ArCv�   ArCQP   ArE�P   CqP#ArE��   ArE�P   ArH3P   Cp��ArHX�   ArH3P   ArJ�P   Cp�	ArJ��   ArJ�P   ArMP   Cp�1ArM:�   ArMP   ArO�P   CpǠArO��   ArO�P   ArQ�P   Cp�ArR�   ArQ�P   ArThP   Cp��ArT��   ArThP   ArV�P   Cp�ArV��   ArV�P   ArYJP   CqM{ArYo�   ArYJP   Ar[�P   Cp��Ar[��   Ar[�P   Ar^,P   Cp�Ar^Q�   Ar^,P   Ar`�P   Cp�Ar`��   Ar`�P   ArcP   Cp��Arc3�   ArcP   AreP   Cp�LAre��   AreP   Arg�P   CqdArh�   Arg�P   ArjaP   Cp�Arj��   ArjaP   Arl�P   Cp� Arl��   Arl�P   AroCP   Cp�Aroh�   AroCP   Arq�P   Cp�kArq��   Arq�P   Art%P   Cp�ArtJ�   Art%P   Arv�P   Cp�Arv��   Arv�P   AryP   CqMAry,�   AryP   Ar{xP   Cq�Ar{��   Ar{xP   Ar}�P   Cq[�Ar~�   Ar}�P   Ar�ZP   Cq'�Ar��   Ar�ZP   Ar��P   Cp�Ar���   Ar��P   Ar�<P   Cp�Ar�a�   Ar�<P   Ar��P   Cpa�Ar���   Ar��P   Ar�P   Cps�Ar�C�   Ar�P   Ar��P   Cp�nAr���   Ar��P   Ar� P   Cp�}Ar�%�   Ar� P   Ar�qP   CqJMAr���   Ar�qP   Ar��P   Cq�Ar��   Ar��P   Ar�SP   Cq-�Ar�x�   Ar�SP   Ar��P   Cq�Ar���   Ar��P   Ar�5P   Cq�Ar�Z�   Ar�5P   Ar��P   Cq2�Ar���   Ar��P   Ar�P   Cp�DAr�<�   Ar�P   Ar��P   Cp�Ar���   Ar��P   Ar��P   CpCDAr��   Ar��P   Ar�jP   Cp�Ar���   Ar�jP   Ar��P   CphAr� �   Ar��P   Ar�LP   Cpm�Ar�q�   Ar�LP   Ar��P   Cq
LAr���   Ar��P   Ar�.P   Cp��Ar�S�   Ar�.P   Ar��P   Cp��Ar���   Ar��P   Ar�P   Cp�Ar�5�   Ar�P   Ar��P   Cp[�Ar���   Ar��P   Ar��P   Cov�Ar��   Ar��P   Ar�cP   Co�%Ar���   Ar�cP   Ar��P   Co�IAr���   Ar��P   Ar�EP   Co�Ar�j�   Ar�EP   ArĶP   Co��Ar���   ArĶP   Ar�'P   Co�6Ar�L�   Ar�'P   ArɘP   Co�hArɽ�   ArɘP   Ar�	P   Co�wAr�.�   Ar�	P   Ar�zP   Co��ArΟ�   Ar�zP   Ar��P   CobtAr��   Ar��P   Ar�\P   Co�GArӁ�   Ar�\P   Ar��P   Co��Ar���   Ar��P   Ar�>P   Co�RAr�c�   Ar�>P   ArگP   Co$Ar���   ArگP   Ar� P   Co?�Ar�E�   Ar� P   ArߑP   CozAr߶�   ArߑP   Ar�P   Co~rAr�'�   Ar�P   Ar�sP   Co�RAr��   Ar�sP   Ar��P   CoZ%Ar�	�   Ar��P   Ar�UP   Coa<Ar�z�   Ar�UP   Ar��P   Cn��Ar���   Ar��P   Ar�7P   Cn��Ar�\�   Ar�7P   Ar�P   CohAAr���   Ar�P   Ar�P   Co6FAr�>�   Ar�P   Ar��P   Cn�:Ar���   Ar��P   Ar��P   Co�Ar� �   Ar��P   Ar�lP   Cn��Ar���   Ar�lP   Ar��P   Co<Ar��   Ar��P   Ar�NP   Cn��Ar�s�   Ar�NP   As�P   Cm��As��   As�P   As0P   Cm��AsU�   As0P   As�P   Cnk'As��   As�P   As	P   CnןAs	7�   As	P   As�P   Cnx�As��   As�P   As�P   Cn�As�   As�P   AseP   Co0RAs��   AseP   As�P   Cn�bAs��   As�P   AsGP   CoO�Asl�   AsGP   As�P   Co|�As��   As�P   As)P   Cn�,AsN�   As)P   As�P   Cn�As��   As�P   AsP   ConAs0�   AsP   As!|P   CoO�As!��   As!|P   As#�P   Co9As$�   As#�P   As&^P   Cn��As&��   As&^P   As(�P   CokBAs(��   As(�P   As+@P   CoPAs+e�   As+@P   As-�P   Co!8As-��   As-�P   As0"P   Cn�1As0G�   As0"P   As2�P   Co=�As2��   As2�P   As5P   Co52As5)�   As5P   As7uP   Co_As7��   As7uP   As9�P   CoN�As:�   As9�P   As<WP   Co�As<|�   As<WP   As>�P   Cn�CAs>��   As>�P   AsA9P   Co?AsA^�   AsA9P   AsC�P   Co��AsC��   AsC�P   AsFP   Co�AsF@�   AsFP   AsH�P   Co�AsH��   AsH�P   AsJ�P   Co�wAsK"�   AsJ�P   AsMnP   Co�FAsM��   AsMnP   AsO�P   Cpm|AsP�   AsO�P   AsRPP   Cp"�AsRu�   AsRPP   AsT�P   Cp�/AsT��   AsT�P   AsW2P   Cp�AsWW�   AsW2P   AsY�P   Cpn�AsY��   AsY�P   As\P   Cp��As\9�   As\P   As^�P   Cp��As^��   As^�P   As`�P   Cps:Asa�   As`�P   AscgP   Cp�Asc��   AscgP   Ase�P   Cp��Ase��   Ase�P   AshIP   Cp�Ashn�   AshIP   Asj�P   Cp�Asj��   Asj�P   Asm+P   Cp�GAsmP�   Asm+P   Aso�P   Cp��Aso��   Aso�P   AsrP   Cp��Asr2�   AsrP   Ast~P   Cq�Ast��   Ast~P   Asv�P   Cq#NAsw�   Asv�P   Asy`P   CpӜAsy��   Asy`P   As{�P   CpޭAs{��   As{�P   As~BP   Cp�As~g�   As~BP   As��P   Cq�As���   As��P   As�$P   Cq!�As�I�   As�$P   As��P   CpΏAs���   As��P   As�P   Cq�As�+�   As�P   As�wP   CqVBAs���   As�wP   As��P   Cq%As��   As��P   As�YP   Cq@"As�~�   As�YP   As��P   Cq�7As���   As��P   As�;P   Cq&As�`�   As�;P   As��P   Cq�]As���   As��P   As�P   Cqn�As�B�   As�P   As��P   Cq�As���   As��P   As��P   Cq�As�$�   As��P   As�pP   CqZ�As���   As�pP   As��P   Cqh�As��   As��P   As�RP   CqB�As�w�   As�RP   As��P   Cq��As���   As��P   As�4P   Cq]�As�Y�   As�4P   As��P   Cq5�As���   As��P   As�P   Cq�As�;�   As�P   As��P   Cq�TAs���   As��P   As��P   Cr�As��   As��P   As�iP   Cq�_As���   As�iP   As��P   Cq�As���   As��P   As�KP   Cq�)As�p�   As�KP   As��P   Cq�.As���   As��P   As�-P   Cq��As�R�   As�-P   AsP   Cqf9As���   AsP   As�P   CqN�As�4�   As�P   AsǀP   Cqv�Asǥ�   AsǀP   As��P   Cq��As��   As��P   As�bP   Cq�Aṡ�   As�bP   As��P   Cq�As���   As��P   As�DP   Cq�;As�i�   As�DP   AsӵP   Cq��As���   AsӵP   As�&P   Cq�uAs�K�   As�&P   AsؗP   Cr�Asؼ�   AsؗP   As�P   Cq�As�-�   As�P   As�yP   Cq��Asݞ�   As�yP   As��P   Cq� As��   As��P   As�[P   CrAs��   As�[P   As��P   Cq��As���   As��P   As�=P   Cq�NAs�b�   As�=P   As�P   Cr�As���   As�P   As�P   Cr(%As�D�   As�P   As�P   Cr�As��   As�P   As�P   Cr�As�&�   As�P   As�rP   Cr�As��   As�rP   As��P   Cq�BAs��   As��P   As�TP   Cq�QAs�y�   As�TP   As��P   CrE�As���   As��P   As�6P   Cq�As�[�   As�6P   As��P   Cr	�As���   As��P   AtP   Cr
DAt=�   AtP   At�P   CqʀAt��   At�P   At�P   CrvAt�   At�P   At	kP   Cr"�At	��   At	kP   At�P   Cq�HAt�   At�P   AtMP   Cr�Atr�   AtMP   At�P   Cq�At��   At�P   At/P   Cq�AtT�   At/P   At�P   Cq݀At��   At�P   AtP   Cq�{At6�   AtP   At�P   Cqr�At��   At�P   At�P   Cq�ZAt�   At�P   AtdP   Cq�At��   AtdP   At!�P   Cq��At!��   At!�P   At$FP   Cq��At$k�   At$FP   At&�P   Cq�XAt&��   At&�P   At)(P   Cq�gAt)M�   At)(P   At+�P   CqҤAt+��   At+�P   At.
P   Cr8�At./�   At.
P   At0{P   Cq��At0��   At0{P   At2�P   CqޥAt3�   At2�P   At5]P   CrE6At5��   At5]P   At7�P   Cq�At7��   At7�P   At:?P   Cq��At:d�   At:?P   At<�P   CqԋAt<��   At<�P   At?!P   Cq��At?F�   At?!P   AtA�P   Cr�AtA��   AtA�P   AtDP   Cq�AtD(�   AtDP   AtFtP   Cq�vAtF��   AtFtP   AtH�P   Cq��AtI
�   AtH�P   AtKVP   Cq�LAtK{�   AtKVP   AtM�P   Cq�1AtM��   AtM�P   AtP8P   Cq��AtP]�   AtP8P   AtR�P   Cq�bAtR��   AtR�P   AtUP   Cq�}AtU?�   AtUP   AtW�P   Cq�AtW��   AtW�P   AtY�P   Cq�QAtZ!�   AtY�P   At\mP   Cq�At\��   At\mP   At^�P   Cq��At_�   At^�P   AtaOP   CrFAtat�   AtaOP   Atc�P   Cq��Atc��   Atc�P   Atf1P   CqyAtfV�   Atf1P   Ath�P   Cr"Ath��   Ath�P   AtkP   Cq�kAtk8�   AtkP   Atm�P   Cq�"Atm��   Atm�P   Ato�P   Cq�Atp�   Ato�P   AtrfP   Cq�Atr��   AtrfP   Att�P   Cq�Att��   Att�P   AtwHP   Cq�=Atwm�   AtwHP   Aty�P   Cq��Aty��   Aty�P   At|*P   Cq�#At|O�   At|*P   At~�P   Cq��At~��   At~�P   At�P   Cq��At�1�   At�P   At�}P   CqyMAt���   At�}P   At��P   CqcZAt��   At��P   At�_P   Cql�At���   At�_P   At��P   CqG�At���   At��P   At�AP   CqN�At�f�   At�AP   At��P   Cq��At���   At��P   At�#P   Cq�XAt�H�   At�#P   At��P   Cq��At���   At��P   At�P   Cq�(At�*�   At�P   At�vP   Cq��At���   At�vP   At��P   Cq��At��   At��P   At�XP   Cq��At�}�   At�XP   At��P   Cq�At���   At��P   At�:P   Cq�At�_�   At�:P   At��P   Cq�?At���   At��P   At�P   Cq�zAt�A�   At�P   At��P   CqƾAt���   At��P   At��P   Cq�OAt�#�   At��P   At�oP   Cq��At���   At�oP   At��P   Cq��At��   At��P   At�QP   Cq�TAt�v�   At�QP   At��P   Cq��At���   At��P   At�3P   Cq�At�X�   At�3P   At��P   CqY�At���   At��P   At�P   CqɹAt�:�   At�P   At��P   CqO�At���   At��P   At��P   Cq�At��   At��P   At�hP   Cr�Atō�   At�hP   At��P   CqBwAt���   At��P   At�JP   Cp�_At�o�   At�JP   At̻P   Cq�At���   At̻P   At�,P   Cq��At�Q�   At�,P   AtѝP   Cq��At���   AtѝP   At�P   Cq��At�3�   At�P   At�P   CqQAt֤�   At�P   At��P   Cq�NAt��   At��P   At�aP   Cq�,Atۆ�   At�aP   At��P   Cq�iAt���   At��P   At�CP   CqSAt�h�   At�CP   At�P   Cq~�At���   At�P   At�%P   CqN�At�J�   At�%P   At�P   Cq΁At��   At�P   At�P   CqN�At�,�   At�P   At�xP   CqMAt��   At�xP   At��P   CqghAt��   At��P   At�ZP   Cq��At��   At�ZP   At��P   Cq�:At���   At��P   At�<P   Cq��At�a�   At�<P   At��P   Cq��At���   At��P   At�P   Cq��At�C�   At�P   At��P   Cq�At���   At��P   Au  P   Cq�Au %�   Au  P   AuqP   Cq�jAu��   AuqP   Au�P   Cq��Au�   Au�P   AuSP   Cq��Aux�   AuSP   Au	�P   CqsbAu	��   Au	�P   Au5P   Cqy�AuZ�   Au5P   Au�P   Cq�$Au��   Au�P   AuP   CqךAu<�   AuP   Au�P   Cq�=Au��   Au�P   Au�P   CqjsAu�   Au�P   AujP   Cr%~Au��   AujP   Au�P   Cq�%Au �   Au�P   AuLP   CqxjAuq�   AuLP   Au�P   Cq�HAu��   Au�P   Au".P   Cq�fAu"S�   Au".P   Au$�P   CqAu$��   Au$�P   Au'P   Cqg<Au'5�   Au'P   Au)�P   Cq�UAu)��   Au)�P   Au+�P   Cq�Au,�   Au+�P   Au.cP   Cq��Au.��   Au.cP   Au0�P   Cq�mAu0��   Au0�P   Au3EP   Cq� Au3j�   Au3EP   Au5�P   Cq�OAu5��   Au5�P   Au8'P   CqͶAu8L�   Au8'P   Au:�P   Cr�Au:��   Au:�P   Au=	P   CrAu=.�   Au=	P   Au?zP   Cq�@Au?��   Au?zP   AuA�P   Cq�ZAuB�   AuA�P   AuD\P   Cq��AuD��   AuD\P   AuF�P   Cq�AuF��   AuF�P   AuI>P   Cq�AuIc�   AuI>P   AuK�P   Cq��AuK��   AuK�P   AuN P   Cq�GAuNE�   AuN P   AuP�P   Cq�AuP��   AuP�P   AuSP   Cq݉AuS'�   AuSP   AuUsP   Cr+�AuU��   AuUsP   AuW�P   Cq�]AuX	�   AuW�P   AuZUP   Cq��AuZz�   AuZUP   Au\�P   Cq��Au\��   Au\�P   Au_7P   Cq�zAu_\�   Au_7P   Aua�P   Cr^Aua��   Aua�P   AudP   Cq�hAud>�   AudP   Auf�P   Cr�Auf��   Auf�P   Auh�P   Cr3�Aui �   Auh�P   AuklP   Cr=Auk��   AuklP   Aum�P   Cr6�Aun�   Aum�P   AupNP   Cq�Aups�   AupNP   Aur�P   Cr Aur��   Aur�P   Auu0P   Cr�AuuU�   Auu0P   Auw�P   Cq�\Auw��   Auw�P   AuzP   Cq҇Auz7�   AuzP   Au|�P   Cq�Au|��   Au|�P   Au~�P   Cq��Au�   Au~�P   Au�eP   Cq��Au���   Au�eP   Au��P   Cq�%Au���   Au��P   Au�GP   Cq�OAu�l�   Au�GP   Au��P   Cr<0Au���   Au��P   Au�)P   Cq��Au�N�   Au�)P   Au��P   Cq��Au���   Au��P   Au�P   CrA�Au�0�   Au�P   Au�|P   Cq�Au���   Au�|P   Au��P   Cq��Au��   Au��P   Au�^P   Cq��Au���   Au�^P   Au��P   Cq�[Au���   Au��P   Au�@P   Cq˘Au�e�   Au�@P   Au��P   Cq��Au���   Au��P   Au�"P   Cr3bAu�G�   Au�"P   Au��P   Cq��Au���   Au��P   Au�P   CrngAu�)�   Au�P   Au�uP   Cr oAu���   Au�uP   Au��P   Cr�Au��   Au��P   Au�WP   CqռAu�|�   Au�WP   Au��P   Cr#Au���   Au��P   Au�9P   Cq�fAu�^�   Au�9P   Au��P   Cq�	Au���   Au��P   Au�P   Cr$Au�@�   Au�P   Au��P   Cr9Au���   Au��P   Au��P   Cq̧Au�"�   Au��P   Au�nP   Cq��Au���   Au�nP   Au��P   CrGLAu��   Au��P   Au�PP   Cq�hAu�u�   Au�PP   Au��P   Cq�BAu���   Au��P   Au�2P   Cr<�Au�W�   Au�2P   AuʣP   Cq��Au���   AuʣP   Au�P   Cq��Au�9�   Au�P   AuυP   Cr?�AuϪ�   AuυP   Au��P   CrAu��   Au��P   Au�gP   Cr;AuԌ�   Au�gP   Au��P   Cr5�Au���   Au��P   Au�IP   Cq��Au�n�   Au�IP   AuۺP   Cr!�Au���   AuۺP   Au�+P   Cq��Au�P�   Au�+P   Au��P   Crk�Au���   Au��P   Au�P   Cr,�Au�2�   Au�P   Au�~P   Cr�Au��   Au�~P   Au��P   Cq��Au��   Au��P   Au�`P   Cr}Au��   Au�`P   Au��P   CruAu���   Au��P   Au�BP   Cr�Au�g�   Au�BP   Au�P   Cr^�