CDF   �   
      time       bnds      lon       lat          1   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       bACCESS-CM2 (2019): 
aerosol: UKCA-GLOMAP-mode
atmos: MetUM-HadGEM3-GA7.1 (N96; 192 x 144 longitude/latitude; 85 levels; top level 85 km)
atmosChem: none
land: CABLE2.5
landIce: none
ocean: ACCESS-OM2 (GFDL-MOM5, tripolar primarily 1deg; 360 x 300 longitude/latitude; 50 levels; top grid cell 0-10 m)
ocnBgchem: none
seaIce: CICE5.1.2 (same grid as ocean)     institution       �CSIRO (Commonwealth Scientific and Industrial Research Organisation, Aspendale, Victoria 3195, Australia), ARCCSS (Australian Research Council Centre of Excellence for Climate System Science)    activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent                    creation_date         2019-11-08T10:12:02Z   data_specs_version        01.00.30   
experiment        )all-forcing simulation of the recent past      experiment_id         
historical     external_variables        	areacella      forcing_index               	frequency         year   further_info_url      Uhttps://furtherinfo.es-doc.org/CMIP6.CSIRO-ARCCSS.ACCESS-CM2.historical.none.r1i1p1f1      grid      ,native atmosphere N96 grid (144x192 latxlon)   
grid_label        gn     history      �Wed Aug 10 15:19:35 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/rlut/ACCESS-CM2_r1i1p1f1/rlut_ACCESS-CM2_r1i1p1f1_historical.mergetime.nc
Fri Apr 08 05:20:05 2022: cdo -O -s -selname,rlut -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rlut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/rlut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.fldmean.nc
Fri Apr 08 05:20:01 2022: cdo -O -s -sellonlatbox,0,360,-90,90 -yearmean -selname,rlut -setmisstoc,0 -remap,global_1,/Users/benjamin/Data/p22c/CMIP6/regrid_weights/rlut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.genbic_weights.nc -setmisstoc,0 -setctomiss,nan /Users/benjamin/Data/p22b/CMIP6/rlut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/rlut/ACCESS-CM2_r1i1p1f1/CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.rlut.gn.v20191108/rlut_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.bic_missto0.yearmean.nc
2019-11-08T10:12:02Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.      initialization_index            institution_id        CSIRO-ARCCSS   mip_era       CMIP6      nominal_resolution        250 km     notes         FExp: CM2-historical; Local ID: bj594; Variable: rlut (['fld_s03i332'])     parent_activity_id        CMIP   parent_experiment_id      	piControl      parent_mip_era        CMIP6      parent_source_id      
ACCESS-CM2     parent_time_units         days since 0950-01-01      parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         atmos      run_variant       jforcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2, N2O, CH4, CFC11, CFC12, CFC113, HCFC22, HFC125, HFC134a)     	source_id         
ACCESS-CM2     source_type       AOGCM      sub_experiment        none   sub_experiment_id         none   table_id      Amon   
table_info        BCreation Date:(30 April 2019) MD5:e14f55f257cceafb2523e41244962371     title         $ACCESS-CM2 output prepared for CMIP6   variable_id       rlut   variant_label         r1i1p1f1   version       	v20191108      cmor_version      3.4.0      tracking_id       1hdl:21.14100/d6208262-0ec5-4683-876b-015693143a5f      license      $CMIP6 model data produced by CSIRO is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (https://creativecommons.org/licenses/).  Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment.  Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file).  The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.   CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      proleptic_gregorian    axis      T               l   	time_bnds                                 t   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               \   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               d   rlut                   	   standard_name         toa_outgoing_longwave_flux     	long_name         TOA Outgoing Longwave Radiation    units         W m-2      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: time: mean   comment       Iat the top of the atmosphere (to be compared with satellite measurements)      cell_measures         area: areacella    history       u2019-11-08T10:12:00Z altered by CMOR: replaced missing value flag (-1.07374e+09) with standard missing value (1e+20).               �                Aq���   Aq��P   Aq�P   Cr[$Aq�6�   Aq�P   Aq��P   Cq��Aq���   Aq��P   Aq��P   Crt�Aq��   Aq��P   Aq�dP   Cr^~Aq���   Aq�dP   Aq��P   Cr��Aq���   Aq��P   Aq�FP   Cr@�Aq�k�   Aq�FP   Aq��P   Cq��Aq���   Aq��P   Aq�(P   Cr1�Aq�M�   Aq�(P   Aq��P   Cr9Aq���   Aq��P   Aq�
P   Cr]WAq�/�   Aq�
P   Aq�{P   Cre0Aq���   Aq�{P   Aq��P   Cq�cAq��   Aq��P   Aq�]P   Cq��AqĂ�   Aq�]P   Aq��P   Cq��Aq���   Aq��P   Aq�?P   Cq��Aq�d�   Aq�?P   Aq˰P   Cq��Aq���   Aq˰P   Aq�!P   Cq�Aq�F�   Aq�!P   AqВP   CrAqз�   AqВP   Aq�P   Cq�|Aq�(�   Aq�P   Aq�tP   Cq��Aqՙ�   Aq�tP   Aq��P   Cq��Aq�
�   Aq��P   Aq�VP   Cq�Aq�{�   Aq�VP   Aq��P   Crg�Aq���   Aq��P   Aq�8P   Cq��Aq�]�   Aq�8P   Aq�P   Cr5AAq���   Aq�P   Aq�P   Cr�Aq�?�   Aq�P   Aq�P   Cq�{Aq��   Aq�P   Aq��P   Cr�Aq�!�   Aq��P   Aq�mP   Cq��Aq��   Aq�mP   Aq��P   Cq�Aq��   Aq��P   Aq�OP   Cq�	Aq�t�   Aq�OP   Aq��P   Cq�Aq���   Aq��P   Aq�1P   Cq��Aq�V�   Aq�1P   Aq��P   Cqh@Aq���   Aq��P   Aq�P   Co�Aq�8�   Aq�P   Aq��P   Cp�FAq���   Aq��P   Aq��P   Cp��Aq��   Aq��P   ArfP   Cp�|Ar��   ArfP   Ar�P   Cq�xAr��   Ar�P   ArHP   Cq{�Arm�   ArHP   Ar�P   Cqy�Ar��   Ar�P   Ar*P   Cq�dArO�   Ar*P   Ar�P   CqHAr��   Ar�P   ArP   Cq|0Ar1�   ArP   Ar}P   Cq�Ar��   Ar}P   Ar�P   Cq�#Ar�   Ar�P   Ar_P   Cq�Ar��   Ar_P   Ar�P   Cr5Ar��   Ar�P   ArAP   Cr�Arf�   ArAP   Ar�P   CrAr��   Ar�P   Ar!#P   Cq�Ar!H�   Ar!#P   Ar#�P   Cq��Ar#��   Ar#�P   Ar&P   Cq�1Ar&*�   Ar&P   Ar(vP   Cp��Ar(��   Ar(vP   Ar*�P   Cq![Ar+�   Ar*�P   Ar-XP   Cq�Ar-}�   Ar-XP   Ar/�P   Cq��Ar/��   Ar/�P   Ar2:P   CqfAr2_�   Ar2:P   Ar4�P   Cq��Ar4��   Ar4�P   Ar7P   Cq!�Ar7A�   Ar7P   Ar9�P   Cq�OAr9��   Ar9�P   Ar;�P   CqdjAr<#�   Ar;�P   Ar>oP   Cq�Ar>��   Ar>oP   Ar@�P   Cp��ArA�   Ar@�P   ArCQP   Cp�nArCv�   ArCQP   ArE�P   Cq
ArE��   ArE�P   ArH3P   Cq��ArHX�   ArH3P   ArJ�P   Cp��ArJ��   ArJ�P   ArMP   CqhArM:�   ArMP   ArO�P   Cqc'ArO��   ArO�P   ArQ�P   Cp�
ArR�   ArQ�P   ArThP   Cq>�ArT��   ArThP   ArV�P   Cp�MArV��   ArV�P   ArYJP   Cq"eArYo�   ArYJP   Ar[�P   CqJ.Ar[��   Ar[�P   Ar^,P   Cq}yAr^Q�   Ar^,P   Ar`�P   Cq�Ar`��   Ar`�P   ArcP   Cq�Arc3�   ArcP   AreP   Cq��Are��   AreP   Arg�P   Cq��Arh�   Arg�P   ArjaP   Cq�yArj��   ArjaP   Arl�P   CqyArl��   Arl�P   AroCP   Cqo:Aroh�   AroCP   Arq�P   Cq��Arq��   Arq�P   Art%P   Cq��ArtJ�   Art%P   Arv�P   Cq��Arv��   Arv�P   AryP   Cr�Ary,�   AryP   Ar{xP   Cq�LAr{��   Ar{xP   Ar}�P   Cq�QAr~�   Ar}�P   Ar�ZP   Cq��Ar��   Ar�ZP   Ar��P   Cq�Ar���   Ar��P   Ar�<P   Cr'�Ar�a�   Ar�<P   Ar��P   Cr�Ar���   Ar��P   Ar�P   Cq�Ar�C�   Ar�P   Ar��P   Cq��Ar���   Ar��P   Ar� P   Cr Ar�%�   Ar� P   Ar�qP   Cq�UAr���   Ar�qP   Ar��P   Cq^�Ar��   Ar��P   Ar�SP   Cqv�Ar�x�   Ar�SP   Ar��P   Cr)Ar���   Ar��P   Ar�5P   Cq2iAr�Z�   Ar�5P   Ar��P   Cq�PAr���   Ar��P   Ar�P   Cq��Ar�<�   Ar�P   Ar��P   Cq�Ar���   Ar��P   Ar��P   Cq�HAr��   Ar��P   Ar�jP   Cq��Ar���   Ar�jP   Ar��P   Cq2VAr� �   Ar��P   Ar�LP   Cq�Ar�q�   Ar�LP   Ar��P   Cq��Ar���   Ar��P   Ar�.P   Cq�Ar�S�   Ar�.P   Ar��P   CqxMAr���   Ar��P   Ar�P   Cq��Ar�5�   Ar�P   Ar��P   Cqk�Ar���   Ar��P   Ar��P   CpA�Ar��   Ar��P   Ar�cP   CpT�Ar���   Ar�cP   Ar��P   CpM�Ar���   Ar��P   Ar�EP   CpM�Ar�j�   Ar�EP   ArĶP   Cp�Ar���   ArĶP   Ar�'P   Cp��Ar�L�   Ar�'P   ArɘP   Cq*oArɽ�   ArɘP   Ar�	P   Cp��Ar�.�   Ar�	P   Ar�zP   Cp�*ArΟ�   Ar�zP   Ar��P   Cp�*Ar��   Ar��P   Ar�\P   Cqo2ArӁ�   Ar�\P   Ar��P   Cp��Ar���   Ar��P   Ar�>P   CpB�Ar�c�   Ar�>P   ArگP   Cp��Ar���   ArگP   Ar� P   Cp��Ar�E�   Ar� P   ArߑP   Cp�Ar߶�   ArߑP   Ar�P   CqAr�'�   Ar�P   Ar�sP   Cp��Ar��   Ar�sP   Ar��P   CpɄAr�	�   Ar��P   Ar�UP   Cp� Ar�z�   Ar�UP   Ar��P   CoԓAr���   Ar��P   Ar�7P   Cp��Ar�\�   Ar�7P   Ar�P   Cp��Ar���   Ar�P   Ar�P   Cq/�Ar�>�   Ar�P   Ar��P   CpY�Ar���   Ar��P   Ar��P   Cp|�Ar� �   Ar��P   Ar�lP   Cp� Ar���   Ar�lP   Ar��P   Cq�Ar��   Ar��P   Ar�NP   CpرAr�s�   Ar�NP   As�P   Cn�GAs��   As�P   As0P   Co��AsU�   As0P   As�P   Cp?�As��   As�P   As	P   Cp`�As	7�   As	P   As�P   Cp�As��   As�P   As�P   Cp�SAs�   As�P   AseP   Cp�eAs��   AseP   As�P   Cp�As��   As�P   AsGP   Cqs�Asl�   AsGP   As�P   Cp�As��   As�P   As)P   Cp�-AsN�   As)P   As�P   Cq1�As��   As�P   AsP   CqAs0�   AsP   As!|P   Cq1�As!��   As!|P   As#�P   Cqi�As$�   As#�P   As&^P   Cp��As&��   As&^P   As(�P   Cp�As(��   As(�P   As+@P   Cp��As+e�   As+@P   As-�P   Cq>�As-��   As-�P   As0"P   CqU�As0G�   As0"P   As2�P   Cq��As2��   As2�P   As5P   Cq"ZAs5)�   As5P   As7uP   Cqy@